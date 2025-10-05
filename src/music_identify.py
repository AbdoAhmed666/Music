import requests
import time
import hmac
import hashlib
import base64
import json
import os

# -----------------------------
# ACRCloud Configuration
# -----------------------------
ACR_HOST = "identify-eu-west-1.acrcloud.com"  # Example: "identify-eu-west-1.acrcloud.com"
ACR_ACCESS_KEY = "b663be4d6e1107c61a13a8b5a968e212"
ACR_ACCESS_SECRET = "qA6fo2l4EHTQsvN9JeUddGBjWTQYdMWXXVLMC2vX"


def identify_with_acr(file_path, offset=0, duration=10):
    """
    Sends an audio clip to ACRCloud and returns the recognition result.
    """
    http_method = "POST"
    http_uri = "/v1/identify"
    data_type = "audio"
    signature_version = "1"
    timestamp = str(int(time.time()))

    # Generate signature
    string_to_sign = f"{http_method}\n{http_uri}\n{ACR_ACCESS_KEY}\n{data_type}\n{signature_version}\n{timestamp}"
    sign = base64.b64encode(
        hmac.new(ACR_ACCESS_SECRET.encode('utf-8'), string_to_sign.encode('utf-8'), digestmod=hashlib.sha1).digest()
    ).decode('utf-8')

    # Read the audio file
    with open(file_path, "rb") as f:
        sample_bytes = f.read()

    files = {'sample': sample_bytes}
    data = {
        'access_key': ACR_ACCESS_KEY,
        'data_type': data_type,
        'signature_version': signature_version,
        'signature': sign,
        'sample_bytes': str(len(sample_bytes)),
        'timestamp': timestamp
    }

    # Send to API
    try:
        r = requests.post("http://" + ACR_HOST + http_uri, files=files, data=data, timeout=10)
        result = r.json()
    except Exception as e:
        raise Exception(f"Failed to connect to ACRCloud: {e}")

    # Format the result
    if "metadata" in result and "music" in result["metadata"]:
        music_info = result["metadata"]["music"][0]
        return {
            "success": True,
            "title": music_info.get("title"),
            "artist": music_info.get("artists", [{}])[0].get("name"),
            "album": music_info.get("album", {}).get("name"),
            "raw": result
        }
    else:
        return {"success": False, "raw": result}
