import logging
import httpx
from typing import Dict, Any, List
from google import genai
from fuse.config import settings
from fuse.plugins.registry import plugin_registry

logger = logging.getLogger(__name__)

def get_provider_from_credential(credential_data: Dict[str, Any]) -> str:
    """Determine the provider from the credential data."""
    provider = (
        credential_data.get("provider") or credential_data.get("type", "unknown")
    ).lower()
    if provider == "ai_provider":
        provider = (
            credential_data.get("data", {}).get("provider", "unknown").lower()
        )
    return provider

async def get_available_models(credential_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Fetch available models for the given credential."""
    provider = get_provider_from_credential(credential_data)

    models = []
    
    # Check plugins for model providers (e.g. Antigravity)
    # This replaces the hardcoded CLIProxy logic
    if not plugin_registry.plugins:
        plugin_registry.initialize()
        
    for plugin in plugin_registry.list_plugins():
        capabilities = plugin.manifest_data.get("capabilities", [])
        # Support both list and dict formats for capabilities
        if isinstance(capabilities, dict):
            # Old format or different schema, check keys? or just skip
            pass 
        elif isinstance(capabilities, list):
            if "ai_provider" in capabilities and plugin.backend_module:
                try:
                    if hasattr(plugin.backend_module, "get_models"):
                        # We might filter by provider if needed, or get all
                         plugin_models = await plugin.backend_module.get_models(credential_data)
                         if plugin_models:
                             models.extend(plugin_models)
                except Exception as e:
                    logger.warning(f"Error fetching models from plugin {plugin.id}: {e}")

    # Google AI (Standard / Non-Proxy)
    if provider == "gemini" or provider == "google_ai":
        # Use genai to list models
        cred_data = credential_data.get("data", {})
        api_key = cred_data.get("api_key")
        access_token = cred_data.get("access_token")
        project_id = cred_data.get("project_id")

        # If we have models from plugins (like Antigravity), we might return them early
        # or combine them. If using Antigravity creds (project_id + access_token) 
        # the plugin likely already handled it. 
        if models and project_id and access_token:
             return models

        try:
            # Standard listing with API Key
            if api_key:
                client = genai.Client(api_key=api_key)
                # list_models returns an iterator
                for m in client.models.list_models():
                    if "generateContent" in m.supported_generation_methods:
                        models.append(
                            {
                                "id": m.name.replace("models/", ""),
                                "label": m.display_name,
                                "provider": "google",
                            }
                        )
            else:
                # Fallback for OAuth - Use public listing or hardcoded if scope issues
                # We try REST API
                async with httpx.AsyncClient() as client:
                    headers = {}
                    if access_token:
                        headers["Authorization"] = f"Bearer {access_token}"
                    elif api_key:
                        headers["x-goog-api-key"] = api_key

                    # Note: v1beta/models usually works public, but let's try
                    resp = await client.get(
                        "https://generativelanguage.googleapis.com/v1beta/models",
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        for m in data.get("models", []):
                            if "generateContent" in m.get(
                                "supportedGenerationMethods", []
                            ):
                                name = m["name"].replace("models/", "")
                                models.append(
                                    {
                                        "id": name,
                                        "label": m.get("displayName", name),
                                        "provider": "google",
                                    }
                                )
        except Exception as e:
            logger.error(f"Failed to fetch Google models: {e}")
            # Fallback list
            return [
                {
                    "id": "gemini-2.0-flash-exp",
                    "label": "Gemini 2.0 Flash (Exp)",
                    "provider": "google",
                },
                {
                    "id": "gemini-1.5-pro-latest",
                    "label": "Gemini 1.5 Pro",
                    "provider": "google",
                },
                {
                    "id": "gemini-1.5-flash-latest",
                    "label": "Gemini 1.5 Flash",
                    "provider": "google",
                },
            ]

    # GitHub Copilot
    elif provider == "github_copilot":
        # Try to fetch dynamically
        copilot_token = credential_data.get("data", {}).get("copilot_token")
        if copilot_token:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        "https://api.githubcopilot.com/models",
                        headers={
                            "Authorization": f"Bearer {copilot_token}",
                            "Editor-Version": "vscode/1.85.0",
                            "User-Agent": "GitHubCopilot/1.138.0",
                        },
                        timeout=5.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        # Expecting OpenAI-format: {"data": [{"id": "gpt-4", ...}]}
                        dynamic_models = []
                        for m in data.get("data", []):
                            # specific filter? or just take all
                            dynamic_models.append(
                                {
                                    "id": m["id"],
                                    "label": f"{m.get('id')} (Copilot)",
                                    "provider": "copilot",
                                }
                            )

                        if dynamic_models:
                            return dynamic_models
                    else:
                        logger.warning(
                            f"Copilot models fetch failed {resp.status_code}: {resp.text}"
                        )
            except Exception as e:
                logger.warning(f"Failed to fetch dynamic Copilot models: {e}")

        # Fallback
        return [
            {"id": "gpt-4", "label": "GPT-4 (Copilot)", "provider": "copilot"},
            {
                "id": "gpt-3.5-turbo",
                "label": "GPT-3.5 Turbo (Copilot)",
                "provider": "copilot",
            },
        ]

    # OpenRouter / OpenAI
    elif provider == "openrouter":
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get("https://openrouter.ai/api/v1/models")
                if resp.status_code == 200:
                    data = resp.json()
                    for m in data.get("data", []):
                        models.append(
                            {
                                "id": m["id"],
                                "label": m.get("name", m["id"]),
                                "provider": "openrouter",
                            }
                        )
                    return models
        except Exception:
            pass

    # Generic Fallback
    if not models:
        models = [
            {"id": "gpt-4o", "label": "GPT-4o", "provider": "openai"},
            {"id": "gpt-4o-mini", "label": "GPT-4o Mini", "provider": "openai"},
            {
                "id": "claude-3-5-sonnet-20240620",
                "label": "Claude 3.5 Sonnet",
                "provider": "anthropic",
            },
        ]

    return models
