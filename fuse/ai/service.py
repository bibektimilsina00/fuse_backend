import json
import logging
import os
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from google import genai
from openai import OpenAI
from fuse.config import settings
from fuse.utils.circuit_breaker import CircuitBreakerOpenError, CircuitBreakers
import httpx
import uuid

logger = logging.getLogger(__name__)
# Import nodes to ensure they are registered
import fuse.workflows.engine.nodes  # noqa

# No longer importing WorkflowNodePublic, WorkflowEdgePublic, NodeData as we use V2 structure in response parsing
from fuse.workflows.engine.nodes.registry import NodeRegistry

from fuse.ai.prompts import FUSE_SYSTEM_PROMPT, WORKFLOW_GENERATION_PROMPT_TEMPLATE
from fuse.ai.models import get_available_models, get_provider_from_credential
from fuse.plugins.registry import plugin_registry


class AIWorkflowService:
    def __init__(self):
        # Initialize AI clients using settings
        self.gemini_api_key = settings.GOOGLE_AI_API_KEY
        self.openai_api_key = settings.OPENAI_API_KEY
        self.anthropic_api_key = settings.ANTHROPIC_API_KEY

        # Initialize models as None
        self.gemini_client = None
        self.openai_client = None
        self.anthropic_client = None
        self.openrouter_client = None

        if self.gemini_api_key:
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)

        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)

        if self.anthropic_api_key:
            self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)

        self.openrouter_api_key = settings.OPENROUTER_API_KEY
        if self.openrouter_api_key:
            self.openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key,
                default_headers={
                    "HTTP-Referer": settings.server_host,
                    "X-Title": settings.PROJECT_NAME,
                },
            )

        # Initialize plugins
        if not plugin_registry.plugins:
            plugin_registry.initialize()

    async def get_available_models(
        self, credential_data: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Fetch available models for the given credential."""
        return await get_available_models(credential_data)

    async def generate_workflow_from_prompt(
        self,
        prompt: str,
        model: str = "openrouter",
        current_nodes: Optional[List[dict]] = None,
        current_edges: Optional[List[dict]] = None,
        credential_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate workflow nodes and edges from natural language prompt"""
        
        # Retrieve only relevant schemas using RAG Service
        # This handles vector search or advanced keyword matching
        from fuse.ai.rag.manager import NodeRAGService
        relevant_schemas = await NodeRAGService.get_instance().retrieve_relevant_nodes(prompt)
        
        system_prompt = self._get_system_prompt(current_nodes, current_edges, schemas=relevant_schemas)
        user_prompt = f"USER REQUEST: {prompt}\n\nPlease generate a workflow JSON based on this request."

        # Determine provider and model from inputs
        provider = "openrouter"
        model_name = model
        
        # If model string contains provider prefix (e.g. "openai/gpt-4"), extract it
        if "/" in model:
            parts = model.split("/", 1)
            provider_candidate = parts[0].lower()
            if provider_candidate in ["openai", "anthropic", "google", "gemini", "copilot", "github_copilot", "openrouter"]:
                provider = provider_candidate
                model_name = parts[1]
        elif model in ["gemini", "google_ai"]:
            provider = "google"
        elif model == "github_copilot":
            provider = "copilot"
        elif model == "openai":
            provider = "openai"
        elif model == "anthropic":
            provider = "anthropic"

        # Credential always overrides provider if present
        if credential_data:
            cred_provider = get_provider_from_credential(credential_data)
            if cred_provider and cred_provider != "unknown":
                provider = cred_provider

        # Normalize provider names
        if provider in ["google_ai", "gemini"]:
            provider = "google"
        elif provider == "github_copilot":
            provider = "copilot"
            
        logger.info(f"AI Generation Request: provider={provider}, model={model_name}")
        print(f"AI Build Prompt ({provider}/{model_name}):\nSYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}", flush=True)

        # Try plugins first (e.g. Antigravity)
        plugin_response = await self._generate_with_plugins(
            system_prompt, user_prompt, model_name, credential_data
        )
        if plugin_response:
            print(f"AI Build Response (plugin):\n{plugin_response}", flush=True)
            return self._parse_ai_response(plugin_response, current_nodes)

        try:
            if provider == "google":
                response_text = await self._generate_with_gemini(
                    system_prompt, user_prompt, model=model_name, credential_data=credential_data
                )
            elif provider == "copilot":
                response_text = await self._generate_with_copilot(
                    system_prompt, user_prompt, model=model_name, credential_data=credential_data
                )
            elif provider == "openai":
                response_text = await self._generate_with_openai(
                    system_prompt, user_prompt, model=model_name if model_name != "openai" else "gpt-4", credential_data=credential_data
                )
            elif provider == "anthropic":
                response_text = await self._generate_with_anthropic(
                    system_prompt, user_prompt, model=model_name if model_name != "anthropic" else "claude-3-sonnet-20240229", credential_data=credential_data
                )
            elif provider == "openrouter":
                response_text = await self._generate_with_openrouter(
                    system_prompt, user_prompt,
                    model=model_name if model_name != "openrouter" else "deepseek/deepseek-r1",
                    credential_data=credential_data,
                )
            else:
                # Default to openrouter if provider is unknown but try using model name directly
                response_text = await self._generate_with_openrouter(
                    system_prompt, user_prompt,
                    model=model,
                    credential_data=credential_data,
                )

            print("\n" + "="*60)
            print(f"🚀 AI BUILD RESPONSE ({provider})")
            print("="*60)
            print(response_text)
            print("="*60 + "\n", flush=True)

            return self._parse_ai_response(response_text, current_nodes)
        except Exception as e:
            print(f"\n❌ AI GENERATION FAILED: {e}", flush=True)
            logger.error(f"AI generation failed: {e}")
            # Fallback to dummy JSON if AI fails during testing
            dummy_file_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "work_flow_eg.json",
            )
            if os.path.exists(dummy_file_path):
                print(f"⚠️ FALLBACK: Using dummy workflow from {dummy_file_path}", flush=True)
                logger.warning("Falling back to dummy workflow example")
                with open(dummy_file_path, "r") as f:
                    response = f.read()
                return self._parse_ai_response(response, current_nodes)
            raise e

    def _get_system_prompt(
        self,
        current_nodes: Optional[List[dict]] = None,
        current_edges: Optional[List[dict]] = None,
        schemas: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        current_workflow_desc = ""
        if current_nodes:
            current_workflow_desc = (
                f"Current workflow has {len(current_nodes)} nodes. Extend it."
            )

        # Get available nodes from registry OR use passed schemas
        if schemas is None:
             schemas = NodeRegistry.get_all_schemas()
             
        nodes_desc = (
            "AVAILABLE NODE TYPES (Use these EXACT names for 'spec.node_name'):\n"
        )

        for schema in schemas:
            try:
                # Reliability: Fallback to 'name' or 'label' if 'id' is missing
                node_id = schema.get("id") or schema.get("name") or schema.get("label")
                if not node_id:
                    continue

                # Input Formatting
                inputs = schema.get("inputs", [])
                inputs_list = []
                for i in inputs:
                    if isinstance(i, dict):
                         i_name = i.get('name', 'param')
                         i_type = i.get('type', 'any')
                         inputs_list.append(f"{i_name} ({i_type})")
                
                inputs_desc = ", ".join(inputs_list)

                # Output Formatting (CRITICAL for wiring)
                outputs = schema.get("outputs", [])
                outputs_list = []
                for o in outputs:
                    if isinstance(o, dict):
                        o_name = o.get('name', 'result')
                        o_type = o.get('type', 'any')
                        outputs_list.append(f"{o_name} ({o_type})")
                
                outputs_desc = ", ".join(outputs_list)
                
                kind = schema.get("type") or schema.get("category") or "action"
                description = schema.get("description", "No description available")
                
                nodes_desc += f"- {node_id} (Kind: {kind})\n"
                nodes_desc += f"  Description: {description}\n"
                if inputs_desc:
                    nodes_desc += f"  Expected Inputs: {inputs_desc}\n"
                if outputs_desc:
                    nodes_desc += f"  Produced Outputs: {outputs_desc}\n"
                nodes_desc += "\n"
            except Exception as e:
                logger.warning(f"Error formatting schema for {schema.get('name')}: {e}")
                continue

        return WORKFLOW_GENERATION_PROMPT_TEMPLATE.format(
            nodes_desc=nodes_desc,
            current_workflow_desc=current_workflow_desc
        )

    async def _generate_with_plugins(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        credential_data: Optional[Dict] = None,
    ) -> Optional[str]:
        """Attempt to generate using plugins."""
        if not plugin_registry.plugins:
            plugin_registry.initialize()

        for plugin in plugin_registry.list_plugins():
            capabilities = plugin.manifest_data.get("capabilities", [])
            # Support both list and dict formats for capabilities
            has_capability = False
            if isinstance(capabilities, dict) and capabilities.get("ai_provider"):
                 has_capability = True
            elif isinstance(capabilities, list) and "ai_provider" in capabilities:
                 has_capability = True
                 
            if has_capability and plugin.backend_module:
                try:
                    if hasattr(plugin.backend_module, "generate"):
                        # We pass system/user prompts separately
                        res = await plugin.backend_module.generate(
                            system_prompt, user_prompt, model, credential_data
                        )
                        if res:
                            return res
                except ValueError as ve:
                    # Specific error (e.g. proxy not running), we might want to surface it if this was the intended target
                    # If the plugin declined (returned None), we wouldn't be here.
                    # If it raised ValueError, it probably accepted but failed.
                    logger.warning(f"Plugin {plugin.id} raised ValueError: {ve}")
                    # If this plugin was the ONLY way to handle this credential, we should probably raise?
                    # But providing a generic fallback is safer.
                    pass
                except Exception as e:
                    logger.warning(f"Plugin {plugin.id} generation failed: {e}")
                    pass
        return None

    async def _generate_with_copilot(
        self, system_prompt: str, user_prompt: str, model: str, credential_data: Dict
    ) -> str:
        """Generate using GitHub Copilot"""
        copilot_token = credential_data.get("data", {}).get("copilot_token")
        if not copilot_token:
            raise ValueError("Copilot token missing.")

        # Use httpx
        headers = {
            "Authorization": f"Bearer {copilot_token}",
            "Editor-Version": "vscode/1.85.0",
            "User-Agent": "GitHubCopilot/1.138.0",
            "Content-Type": "application/json",
        }

        # Parse model or default
        copilot_model = model
        if not copilot_model or "/" in copilot_model:
            # Strip provider prefix if present (e.g. "copilot/gpt-4o")
            if "/" in model:
                copilot_model = model.split("/")[-1]
            else:
                copilot_model = "gpt-4"

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": user_prompt},
            ],
            "model": copilot_model,
            "temperature": 0.7,
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.githubcopilot.com/chat/completions",
                headers=headers,
                json=payload,
                timeout=60.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def _generate_with_gemini(
        self, system_prompt: str, user_prompt: str, model: str = "gemini-2.0-flash", credential_data: Optional[Dict] = None
    ) -> str:
        """Generate using Google Gemini (Direct API)"""
        client = self.gemini_client
        if credential_data:
            api_key = credential_data.get("data", {}).get("api_key")
            if api_key:
                client = genai.Client(api_key=api_key)

        if not client:
            raise ValueError(
                "Gemini API key not configured. Please set GOOGLE_AI_API_KEY environment variable or provide a credential."
            )
        async with CircuitBreakers.google():
            response = client.models.generate_content(
                model=model, contents=f"{system_prompt}\n\n{user_prompt}"
            )
            return response.text or ""

    async def _generate_with_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "gpt-4",
        credential_data: Optional[Dict] = None,
    ) -> str:
        """Generate using OpenAI GPT"""
        client = self.openai_client
        if credential_data:
            api_key = credential_data.get("data", {}).get("api_key")
            base_url = credential_data.get("data", {}).get("base_url")
            if api_key:
                client = OpenAI(api_key=api_key, base_url=base_url)

        if not client:
            raise ValueError(
                "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable or provide a credential."
            )
        async with CircuitBreakers.openai():
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content

    async def _generate_with_anthropic(
        self, system_prompt: str, user_prompt: str, model: str = "claude-3-sonnet-20240229", credential_data: Optional[Dict] = None
    ) -> str:
        """Generate using Anthropic Claude"""
        client = self.anthropic_client
        if credential_data:
            api_key = credential_data.get("data", {}).get("api_key")
            if api_key:
                client = Anthropic(api_key=api_key)

        if not client:
            raise ValueError(
                "Anthropic API key not configured. Please set ANTHROPIC_API_KEY environment variable or provide a credential."
            )
        async with CircuitBreakers.anthropic():
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text

    async def _generate_with_openrouter(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "deepseek/deepseek-r1",
        credential_data: Optional[Dict] = None,
    ) -> str:
        """Generate using OpenRouter"""
        client = self.openrouter_client
        if credential_data:
            api_key = credential_data.get("data", {}).get("api_key")
            if api_key:
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                )

        if not client:
            raise ValueError(
                "OpenRouter API key not configured. Please set OPENROUTER_API_KEY environment variable or provide a credential."
            )

        # Use a generic HTTP circuit breaker for OpenRouter
        async with CircuitBreakers.http("openrouter-api"):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=4096,
                extra_body={"provider": {"allow_fallbacks": False}},
            )
            return response.choices[0].message.content

    def _parse_ai_response(
        self, response: str, current_nodes: Optional[List[dict]] = None
    ) -> Dict[str, Any]:
        """Parse AI response and extract JSON workflow. Returns V2 structure."""
        try:
            # Clean response more robustly
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[-1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[-1].split("```")[0].strip()

            json_start = json_str.find("{")
            json_end = json_str.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = json_str[json_start:json_end]
                workflow_data = json.loads(json_str)

                # Ensure it has the V2 top-level keys
                required_keys = ["meta", "graph", "execution", "observability", "ai"]
                for key in required_keys:
                    if key not in workflow_data:
                        # Add default if missing (graceful fallback)
                        if key == "meta":
                            workflow_data[key] = {"name": "AI Generated Workflow"}
                        elif key == "graph":
                            workflow_data[key] = {"nodes": [], "edges": []}
                        elif key == "execution":
                            workflow_data[key] = {"mode": "async"}
                        elif key == "observability":
                            workflow_data[key] = {"logging": True}
                        elif key == "ai":
                            workflow_data[key] = {"generated_by": "workflow-llm"}

                # Normalize observability fields to ensure they're simple booleans
                if "observability" in workflow_data:
                    obs = workflow_data["observability"]
                    # Convert complex objects to simple booleans
                    for field in ["logging", "metrics", "tracing"]:
                        if field in obs:
                            val = obs[field]
                            # If it's a dict/object, extract 'enabled' or just set to True
                            if isinstance(val, dict):
                                obs[field] = val.get("enabled", True)
                            # Ensure it's a boolean
                            elif not isinstance(val, bool):
                                obs[field] = bool(val)


                # Extract nodes and edges for frontend compatibility
                nodes = workflow_data.get("graph", {}).get("nodes", [])
                edges = workflow_data.get("graph", {}).get("edges", [])

                return {
                    "nodes": nodes,  # Frontend expects this at top level
                    "edges": edges,  # Frontend expects this at top level
                    "suggestions": [
                        "Configure credential fields",
                        "Test workflow execution",
                    ],
                }

            raise ValueError(
                f"No JSON object found in AI response. Raw string: {response[:200]}..."
            )

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.error(f"Extraction failed for: {json_str[:200]}...")
            raise ValueError(f"Invalid JSON in AI response: {e}")

    async def execute_node(
        self, config: Dict[str, Any], input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute an AI node (Legacy/Simple). New nodes should use call_llm."""
        prompt_template = config.get("prompt", "")
        model = config.get("model", "gemini")
        system_instruction = config.get(
            "system_instruction", "You are a helpful assistant."
        )

        # Simple variable substitution
        prompt = prompt_template
        for key, value in input_data.items():
            if isinstance(value, (str, int, float, bool)):
                prompt = prompt.replace(f"{{{{{key}}}}}", str(value))

        full_prompt = f"{system_instruction}\n\n{prompt}"

        try:
            if model == "gemini":
                response = await self._generate_with_gemini(full_prompt)
            elif model == "openai":
                response = await self._generate_with_openai(full_prompt)
            elif model == "anthropic":
                response = await self._generate_with_anthropic(full_prompt)
            else:
                response = await self._generate_with_openrouter(
                    full_prompt, model=model
                )

            return {"response": response}
        except Exception as e:
            logger.error(f"AI execution error: {e}")
            raise e

    def _get_node_context_for_chat(self) -> str:
        """Get a concise summary of all registered node types with their inputs for chat context."""
        schemas = NodeRegistry.get_all_schemas()
        
        lines = [
            "AVAILABLE NODE TYPES:",
            "When helping users, you can recommend using these specific nodes if they fit the task:",
        ]
        
        for schema in schemas:
            try:
                name = schema.get("name") or schema.get("id")
                desc = schema.get("description", "No description")
                kind = schema.get("type", "unknown")
                
                # Basic info
                line = f"- **{name}** ({kind}): {desc}"
                
                # Inputs for context
                inputs = schema.get("inputs", [])
                if inputs:
                    input_names = [i.get("name") for i in inputs if isinstance(i, dict) and "name" in i]
                    if input_names:
                        line += f" [Inputs: {', '.join(input_names)}]"
                        
                lines.append(line)
            except Exception:
                continue
                
        return "\n".join(lines)

    async def call_llm(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        credential: Dict[str, Any] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Unified method to call LLMs with dynamic credentials and structured parameters.
        Returns standardized dict with 'content' and 'usage'.
        """
        if not messages:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if user_prompt:
                messages.append({"role": "user", "content": user_prompt})

        # Determine provider
        provider = (
            credential.get("provider") or credential.get("type", "unknown")
        ).lower()
        if provider == "ai_provider":
            provider = credential.get("data", {}).get("provider", "unknown").lower()

        # Determine the user prompt for RAG context
        # It might be passed as arg OR be the last message in messages list
        current_user_prompt = user_prompt
        if not current_user_prompt and messages:
            last_msg = messages[-1]
            if last_msg.get("role") == "user":
                current_user_prompt = last_msg.get("content")
        
        # Inject Fuse System Prompt if not already present or as a prefix
        # We also want to inject the list of available nodes so the AI knows what's installed
        # Update: We now use RAG to fetch only RELEVANT nodes if a user prompt is present
        node_context = ""
        if current_user_prompt:
             # Use RAG to get relevant nodes
             # We need to import inside method to avoid circular imports if any
            try:
                from fuse.ai.rag.manager import NodeRAGService
                # Limit to top 20 relevant nodes for chat context
                rag_service = NodeRAGService.get_instance()
                # Debug: ensure index size
                # logger.info(f"RAG Index Status: last_indexed={rag_service.last_indexed}")
                
                # If user is asking for "all nodes" or "available nodes", expand limit to include everything
                # Or bypass RAG if the intent is to see the full library
                limit_val = 20
                if any(word in current_user_prompt.lower() for word in ["all", "every", "complete list", "available nodes"]):
                    limit_val = 100 # Large enough for the current library
                
                relevant_schemas = await rag_service.retrieve_relevant_nodes(current_user_prompt, limit=limit_val)
                # print(f"RAG Search '{current_user_prompt}' found {len(relevant_schemas)} nodes.", flush=True)
                
                if relevant_schemas:
                    lines = [
                        "RELEVANT NODE TYPES (RAG Retrieval):",
                        "The following nodes seem relevant to the user's request. You can recommend them:",
                    ]
                    for schema in relevant_schemas:
                        name = schema.get("name") or schema.get("id")
                        desc = schema.get("description", "No description")
                        # "type" might be missing or None. Fallback to "category" or check explicit keys.
                        kind = schema.get("type") or schema.get("category") or "action"
                        
                         # Basic info
                        line = f"- **{schema.get('label', name)}** ({kind}): {desc}"
                        # Inputs for context
                        inputs = schema.get("inputs", [])
                        if inputs:
                            input_names = [i.get("name") for i in inputs if isinstance(i, dict) and "name" in i]
                            if input_names:
                                line += f" [Inputs: {', '.join(input_names)}]"
                        lines.append(line)
                    node_context = "\n".join(lines)
            except Exception as e:
                logger.warning(f"RAG retrieval failed for chat: {e}")
                # Fallback handled below

        # Fallback: If no context from RAG (or no prompt), provide all nodes
        # UPDATE: User requested to FORCE use of RAG and NOT fallback to full list
        # if not node_context:
        #      node_context = self._get_node_context_for_chat()
        
        final_messages = []
        has_system = False
        for msg in messages:
            if msg["role"] == "system":
                # Prepend our context to user's system prompt
                new_content = f"{FUSE_SYSTEM_PROMPT}\n\n{node_context}\n\nAdditional Instructions:\n{msg['content']}"
                final_messages.append({"role": "system", "content": new_content})
                has_system = True
            else:
                final_messages.append(msg)
        
        if not has_system:
            final_messages.insert(0, {"role": "system", "content": f"{FUSE_SYSTEM_PROMPT}\n\n{node_context}"})
        
        # Update messages reference for all providers
        messages = final_messages

        # Log the full prompt for debugging/transparency as requested
        print(f"AI Prompt ({provider}/{model}):\n{json.dumps(messages, indent=2)}", flush=True)

        cred_data = credential.get("data", {})
        api_key = cred_data.get("api_key")
        base_url = cred_data.get("base_url")

        # =========================================================================
        # Google AI (OAuth or API Key)
        # =========================================================================
        if provider == "gemini" or provider == "google_ai":
            access_token = cred_data.get("access_token")

            # Combine formatting for Gemini
            combined_prompt = ""
            for msg in messages:
                combined_prompt += f"{msg['role'].upper()}: {msg['content']}\n\n"

            # 1. OAuth Access Token (Google AI Login) - Using CLIProxyAPI
            # CLIProxyAPI handles OAuth token refresh, request formatting, and endpoint fallback
            if access_token:
                try:
                    # Ensure CLIProxyAPI is running - DO NOT auto-start from here
                    # User should start it from the Plugins page
                    if not await self._is_proxy_running():
                        raise ValueError("Google AI Plugin (Antigravity) is not running. Please start it from the Plugins page.")
                    
                    target_model = model or "gemini-3-pro-preview"
                    if "/" in target_model:
                        target_model = target_model.split("/")[-1]
                    
                    # Map model names to CLIProxyAPI model names
                    # CLIProxyAPI uses "gemini-" prefix for Claude models via Antigravity
                    model_mapping = {
                        "claude-sonnet-4-5": "gemini-claude-sonnet-4-5",
                        "claude-sonnet-4-5-thinking": "gemini-claude-sonnet-4-5-thinking",
                        "claude-opus-4-5-thinking": "gemini-claude-opus-4-5-thinking",
                        "gemini-3-pro": "gemini-3-pro-preview",
                        "gemini-3-flash": "gemini-3-flash-preview",
                        "gpt-4o": "gemini-3-pro-preview", # Fallback to best avail
                        "gpt-4o-mini": "gemini-2.0-flash", # Fallback to fast avail
                        "openai/gpt-4o-mini": "gemini-2.0-flash",
                    }
                    proxy_model = model_mapping.get(target_model, target_model)
                    
                    # If still unmapped and looks like GPT, default to a supported one
                    if "gpt" in proxy_model and proxy_model not in model_mapping.values():
                         proxy_model = "gemini-3-pro-preview" 
                    

                    
                    logger.info(f"CLIProxyAPI request: model={proxy_model}")
                    
                    # OpenAI-compatible request to CLIProxyAPI
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        resp = await client.post(
                            f"{CLIPROXY_URL}/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {CLIPROXY_API_KEY}",
                                "Content-Type": "application/json",
                            },
                            json={
                                "model": proxy_model,
                                "messages": messages,
                                "max_tokens": max_tokens,
                                "temperature": temperature,
                            },
                        )
                        
                        if resp.status_code != 200:
                            error_text = resp.text
                            logger.error(f"CLIProxyAPI error ({resp.status_code}): {error_text[:300]}")
                            raise ValueError(f"CLIProxyAPI error: {error_text[:200]}")
                        
                        resp_json = resp.json()
                        content = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                        print("\n" + "="*50)
                        print(f"🗨️  AI RESPONSE ({provider}/{proxy_model})")
                        print("="*50)
                        print(content)
                        print("="*50 + "\n", flush=True)
                        usage = resp_json.get("usage", {})
                        
                        return {
                            "content": content,
                            "usage": {
                                "prompt_tokens": usage.get("prompt_tokens", 0),
                                "completion_tokens": usage.get("completion_tokens", 0),
                                "total_tokens": usage.get("total_tokens", 0),
                            },
                        }

                except Exception as e:
                    logger.error(f"CLIProxyAPI Failed: {e}")
                    raise e

            # 2. Existing API Key Flow
            client = genai.Client(api_key=api_key) if api_key else self.gemini_client
            if not client:
                raise ValueError(
                    "Gemini API key not found and no OAuth token provided."
                )

            response = client.models.generate_content(
                model=model or "gemini-2.0-flash",
                contents=combined_prompt,
                config={"temperature": temperature, "max_output_tokens": max_tokens},
            )
            print("\n" + "="*50)
            print(f"🗨️  AI RESPONSE ({provider}/{model})")
            print("="*50)
            print(response.text)
            print("="*50 + "\n", flush=True)
            return {
                "content": response.text,
                "usage": {
                    "prompt_tokens": (
                        response.usage_metadata.prompt_token_count
                        if response.usage_metadata
                        else 0
                    ),
                    "completion_tokens": (
                        response.usage_metadata.candidates_token_count
                        if response.usage_metadata
                        else 0
                    ),
                    "total_tokens": (
                        response.usage_metadata.total_token_count
                        if response.usage_metadata
                        else 0
                    ),
                },
            }

        # =========================================================================
        # GitHub Copilot
        # =========================================================================
        elif provider == "github_copilot":
            copilot_token = cred_data.get("copilot_token")
            if not copilot_token:
                raise ValueError(
                    "GitHub Copilot token missing. Please re-authenticate."
                )

            headers = {
                "Authorization": f"Bearer {copilot_token}",
                "Editor-Version": "vscode/1.85.0",
                "User-Agent": "GitHubCopilot/1.138.0",
                "Content-Type": "application/json",
            }

            # Map standard model names to Copilot internal names?
            # Copilot usually supports 'gpt-4', 'gpt-3.5-turbo'.
            # If user sends 'openai/gpt-4', split it.
            copilot_model = model
            if "/" in model:
                copilot_model = model.split("/")[-1]

            # Copilot often defaults to specific models if not specified or different names.
            # But the API is OpenAI compatible.

            payload = {
                "messages": messages,
                "model": copilot_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.githubcopilot.com/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60.0,
                )

                if resp.status_code != 200:
                    raise ValueError(
                        f"GitHub Copilot API Error ({resp.status_code}): {resp.text}"
                    )

                resp_json = resp.json()
                content = resp_json["choices"][0]["message"]["content"]
                print("\n" + "="*50)
                print(f"🗨️  AI RESPONSE ({provider}/{copilot_model})")
                print("="*50)
                print(content)
                print("="*50 + "\n", flush=True)
                return {
                    "content": content,
                    "usage": resp_json.get("usage", {}),
                }

        elif provider == "openai" or (provider == "openrouter" and not base_url):
            # ... (existing OpenAI logic) ...
            actual_base_url = base_url or (
                "https://openrouter.ai/api/v1" if provider == "openrouter" else None
            )
            client = (
                OpenAI(api_key=api_key, base_url=actual_base_url)
                if api_key
                else self.openai_client
            )
            if not client:
                raise ValueError(f"{provider.capitalize()} API key not found.")

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            ai_content = response.choices[0].message.content
            print("\n" + "="*50)
            print(f"🗨️  AI RESPONSE ({provider}/{model})")
            print("="*50)
            print(ai_content)
            print("="*50 + "\n", flush=True)
            return {
                "content": ai_content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }

        elif provider == "anthropic":
            client = Anthropic(api_key=api_key) if api_key else self.anthropic_client
            if not client:
                raise ValueError("Anthropic API key not found.")

            # Extract system message for Anthropic
            sys_msg = next(
                (m["content"] for m in messages if m["role"] == "system"), None
            )
            filtered_messages = [m for m in messages if m["role"] != "system"]

            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=sys_msg,
                messages=filtered_messages,
                temperature=temperature,
            )
            ai_content = response.content[0].text
            print("\n" + "="*50)
            print(f"🗨️  AI RESPONSE ({provider}/{model})")
            print("="*50)
            print(ai_content)
            print("="*50 + "\n", flush=True)
            return {
                "content": ai_content,
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens
                    + response.usage.output_tokens,
                },
            }

        elif provider == "ai_provider" or base_url:
           
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            ai_content = response.choices[0].message.content
            print("\n" + "="*50)
            print(f"🗨️  AI RESPONSE ({provider}/{model})")
            print("="*50)
            print(ai_content)
            print("="*50 + "\n", flush=True)
            return {
                "content": ai_content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }

        else:
            raise ValueError(f"Unsupported AI provider: {provider}")


# Singleton instance
ai_service = AIWorkflowService()
