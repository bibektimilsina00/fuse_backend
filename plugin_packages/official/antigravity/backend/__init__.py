from . import manager
import httpx
import logging
from typing import List, Dict, Any, Optional


logger = logging.getLogger(__name__)

# --- Lifecycle Management ---

def get_status(manifest: dict) -> dict:
    """
    Get the dynamic status of the plugin combined with manifest data.
    """
    proxy_status = manager.get_cliproxy_status()
    installed = proxy_status["installed"]
    running = proxy_status["running"]
    
    status_str = "not_installed"
    if installed:
        status_str = "active" if running else "installed"
        
    return {
        **manifest,
        "installed": installed,
        "running": running,
        "status": status_str,
        "details": proxy_status
    }

def install() -> bool:
    return manager.download_cliproxy()

def start() -> bool:
    if manager.is_cliproxy_running():
        return True
    return manager.start_cliproxy()

def stop() -> bool:
    manager.stop_cliproxy()
    return True

def uninstall() -> bool:
    return manager.uninstall_cliproxy()

def perform_custom_action(action: str, **kwargs) -> dict:
    if action == "login":
        success = manager.run_antigravity_login()
        if not success:
             raise Exception("Login failed")
        return {"success": True, "message": "Login successful"}
    
    raise ValueError(f"Unknown custom action: {action}")

# --- AI Provider Implementation ---

CLIPROXY_URL = "http://127.0.0.1:8317"
CLIPROXY_API_KEY = "fuse-local-dev-key"

async def is_available() -> bool:
    """Check if the plugin's provider service is available."""
    return manager.is_cliproxy_running()

async def get_models(credential_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Fetch available models from the proxy."""
    if not await is_available():
        return []

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{CLIPROXY_URL}/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                model_list = data.get("data", []) if isinstance(data, dict) else data
                
                dynamic_models = []
                for m in model_list:
                    mid = m.get("id")
                    if mid:
                        model_provider = "google"
                        if "claude" in mid:
                            model_provider = "anthropic"
                        elif "gpt" in mid:
                            model_provider = "openai"
                            
                        dynamic_models.append({
                            "id": mid,
                            "label": mid, 
                            "provider": model_provider, 
                            "description": f"Managed via Antigravity" 
                        })
                return dynamic_models
    except Exception as e:
        logger.warning(f"Failed to fetch dynamic models from Antigravity: {e}")
    
    # Fallback to static list known to contain common proxied models
    return [
        {"id": "gemini-3-pro-preview", "label": "Gemini 3 Pro", "provider": "google"},
        {"id": "gemini-3-flash-preview", "label": "Gemini 3 Flash", "provider": "google"},
        {"id": "gemini-2.5-flash", "label": "Gemini 2.5 Flash", "provider": "google"},
        {"id": "claude-sonnet-4-5", "label": "Claude Sonnet 4.5", "provider": "anthropic"},
        {"id": "claude-sonnet-4-5-thinking", "label": "Claude Sonnet 4.5 Thinking", "provider": "anthropic"},
        {"id": "claude-opus-4-5-thinking", "label": "Claude Opus 4.5 Thinking", "provider": "anthropic"},
        {"id": "gpt-oss-120b-medium", "label": "GPT-OSS 120B Medium", "provider": "openai"},
    ]

async def generate(system_prompt: str, user_prompt: str, model: str, credential_data: Optional[Dict] = None) -> Optional[str]:
    """Generate content using the proxy."""
    # Check if this request is intended for Antigravity
    # We rely on the presence of an access_token (Google OAuth) 
    # as the signal to use the proxy for managed models.
    access_token = None
    if credential_data:
        access_token = credential_data.get("data", {}).get("access_token")
    
    if not access_token:
        # We also check if the plugin is explicitly configured/overridden? 
        # For now, match legacy behavior: only use proxy if access_token is present.
        return None

    if not await is_available():
        # If we have an access token (intent to use proxy) but it's not running, we error or auto-start?
        # Legacy code raised ValueError.
        raise ValueError("Antigravity proxy is not running. Please start it from the Plugins page.")
        
    proxy_model = model if model else "gemini-3-pro-preview"
    
    logger.info(f"Antigravity generation request: model={proxy_model}")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{CLIPROXY_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {CLIPROXY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": proxy_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            },
        )
        
        if resp.status_code == 200:
            return resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            raise Exception(f"Antigravity generation failed ({resp.status_code}): {resp.text}")
