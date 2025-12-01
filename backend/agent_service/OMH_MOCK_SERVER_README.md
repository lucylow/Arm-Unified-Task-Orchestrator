# OMH Mock Server

A FastAPI-based mock server for testing Open Mobile Hub (OMH) OAuth2 authentication, user profiles, and location services with AutoRL.

## Features

✅ **OAuth2 Authentication Endpoints**
- Password grant flow
- Refresh token flow
- JWT bearer assertion (OMH token exchange)
- Mock access tokens, refresh tokens, and ID tokens

✅ **User Profile Management**
- Get authenticated user profile
- List users with filtering
- Get user by ID

✅ **OMH Maps/Location Services**
- Mock location data for multiple cities
- Location selection simulation
- Place ID lookup

✅ **Development Tools**
- Token listing endpoint
- Health check
- Interactive API docs (Swagger UI)

## Quick Start

### 1. Install Dependencies

```bash
cd src/agent_service
pip install -r omh_mock_requirements.txt
```

### 2. Run the Server

```bash
python omh_mock_server.py
```

The server will start on `http://localhost:8001`

### 3. View API Documentation

Open your browser to:
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

## Usage Examples

### Get Mock Tokens

```bash
curl http://localhost:8001/mock/tokens
```

### Authenticate with Username/Password

```bash
curl -X POST http://localhost:8001/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "grant_type": "password",
    "username": "alice.smith",
    "password": "any_password"
  }'
```

Response:
```json
{
  "access_token": "mock_access_token_abc123xyz456",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "mock_refresh_token_xyz789abc123",
  "scope": "openid profile email",
  "id_token": "mock_id_token_jwt_payload"
}
```

### Get User Profile (Authenticated)

```bash
curl -H "Authorization: Bearer mock_access_token_abc123xyz456" \
  http://localhost:8001/api/v1/user/profile
```

Response:
```json
{
  "user_id": "user_1001",
  "username": "alice.smith",
  "email": "alice.smith@example.com",
  "full_name": "Alice Smith",
  "roles": ["user", "tester"],
  "profile_picture_url": "https://example.com/profiles/alice.jpg",
  "account_status": "active",
  "last_login": "2025-10-10T10:00:00Z"
}
```

### OMH Token Exchange

```bash
curl -X POST http://localhost:8001/mobile/platform/sso/exchange-token \
  -H "Authorization: Bearer external-jwt-token" \
  -H "oracle-mobile-backend-id: YOUR_BACKEND_ID"
```

### Get Location Data

```bash
curl -H "Authorization: Bearer mock_access_token_abc123xyz456" \
  http://localhost:8001/api/v1/maps/location
```

Response:
```json
{
  "location": {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "address": "San Francisco, CA, USA",
    "place_id": "mock_place_12345"
  }
}
```

### List All Locations

```bash
curl -H "Authorization: Bearer mock_access_token_abc123xyz456" \
  http://localhost:8001/api/v1/maps/locations
```

## Available Mock Users

| Username | User ID | Email | Roles | Status |
|----------|---------|-------|-------|--------|
| alice.smith | user_1001 | alice.smith@example.com | user, tester | active |
| bob.jones | user_1002 | bob.jones@example.com | admin | active |
| carol.wu | user_1003 | carol.wu@example.com | user | inactive |
| testuser | user_001 | testuser@example.com | user, beta_tester | active |

## Available Mock Tokens

For quick testing, use these pre-configured tokens:

```
Alice (basic): mock_access_token_abc123xyz456
Alice (JWT): eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJVc2VyMTIzIiwibmFtZSI6IkFsaWNlIFNtaXRoIiwiYWRtaW4iOnRydWUsImlhdCI6MTY5NjQ2ODAwMCwiZXhwIjoxNjk2NDcxNjAwfQ._sVTeECHBI_sdMjXSQ7Vow_index53jZnZ1049k
Test User: mock_access_token_123456789
```

## API Endpoints

### Authentication
- `POST /auth/token` - Get access token (password, refresh_token, JWT bearer)
- `POST /auth/refresh` - Refresh access token
- `POST /mobile/platform/sso/exchange-token` - OMH SSO token exchange

### User Management
- `GET /api/v1/user/profile` - Get authenticated user profile
- `GET /api/v1/users` - List all users (with optional status filter)
- `GET /api/v1/users/{user_id}` - Get user by ID

### Maps/Location
- `GET /api/v1/maps/location` - Get location (default or by place_id)
- `GET /api/v1/maps/locations` - List all available locations
- `POST /api/v1/maps/select-location` - Select/save a location

### System
- `GET /` - API info and endpoint listing
- `GET /health` - Health check
- `GET /mock/tokens` - Get mock tokens for development

## Integration with AutoRL

### Frontend Integration (React/JavaScript)

```javascript
const API_BASE = 'http://localhost:8001';

// Login and get token
async function login(username, password) {
  const response = await fetch(`${API_BASE}/auth/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      grant_type: 'password',
      username,
      password
    })
  });
  const data = await response.json();
  localStorage.setItem('access_token', data.access_token);
  return data;
}

// Get user profile
async function getUserProfile() {
  const token = localStorage.getItem('access_token');
  const response = await fetch(`${API_BASE}/api/v1/user/profile`, {
    headers: { 'Authorization': `Bearer ${token}` }
  });
  return response.json();
}

// Get location
async function getLocation() {
  const token = localStorage.getItem('access_token');
  const response = await fetch(`${API_BASE}/api/v1/maps/location`, {
    headers: { 'Authorization': `Bearer ${token}` }
  });
  return response.json();
}
```

### Android/Kotlin Integration

```kotlin
// MockTokenProvider.kt
object MockTokenProvider {
    const val BASE_URL = "http://10.0.2.2:8001" // For Android emulator
    const val ACCESS_TOKEN = "mock_access_token_abc123xyz456"
}

// Retrofit AuthInterceptor
class AuthInterceptor : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        val original = chain.request()
        val request = original.newBuilder()
            .header("Authorization", "Bearer ${MockTokenProvider.ACCESS_TOKEN}")
            .method(original.method, original.body)
            .build()
        return chain.proceed(request)
    }
}
```

### Python/Backend Integration

```python
import requests

API_BASE = "http://localhost:8001"

# Get token
response = requests.post(f"{API_BASE}/auth/token", json={
    "grant_type": "password",
    "username": "alice.smith",
    "password": "test"
})
token_data = response.json()
access_token = token_data["access_token"]

# Use token
headers = {"Authorization": f"Bearer {access_token}"}
profile = requests.get(f"{API_BASE}/api/v1/user/profile", headers=headers).json()
print(profile)
```

## Configuration

### Custom Port

```bash
python omh_mock_server.py --port 9000
```

Or modify `omh_mock_server.py`:
```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
```

### Custom Mock Data

Edit `mock_data/omh_mock_data.json` to add your own users, tokens, or locations.

## Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY omh_mock_requirements.txt .
RUN pip install --no-cache-dir -r omh_mock_requirements.txt

COPY omh_mock_server.py .
COPY mock_data/ mock_data/

EXPOSE 8001

CMD ["python", "omh_mock_server.py"]
```

Build and run:

```bash
docker build -t omh-mock-server .
docker run -p 8001:8001 omh-mock-server
```

## Testing

### Postman Collection

Import the OpenAPI spec from `http://localhost:8001/openapi.json` into Postman for a complete collection of endpoints.

### curl Test Script

```bash
#!/bin/bash

# Get token
TOKEN=$(curl -s -X POST http://localhost:8001/auth/token \
  -H "Content-Type: application/json" \
  -d '{"grant_type":"password","username":"alice.smith","password":"test"}' \
  | jq -r '.access_token')

echo "Token: $TOKEN"

# Get profile
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8001/api/v1/user/profile | jq

# Get location
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8001/api/v1/maps/location | jq
```

## Troubleshooting

### Port Already in Use

Change the port in the server startup:
```python
uvicorn.run(app, host="0.0.0.0", port=8002)
```

### CORS Issues

The server has CORS enabled for all origins. For production, restrict in `omh_mock_server.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    ...
)
```

### Token Not Found

Check available tokens at: http://localhost:8001/mock/tokens

## Production Notes

⚠️ **This is a mock server for development/testing only!**

- Tokens are static and don't expire
- No actual authentication validation
- No database or persistence
- Not secure for production use

For production OMH integration, refer to:
- [OMH Authentication Docs](https://github.com/openmobilehub/omh-auth)
- [Oracle Mobile Hub Docs](https://docs.oracle.com/en/cloud/paas/mobile-hub/)

## License

MIT License - Use freely for testing and development.

## Support

For issues or questions:
- Check the interactive docs at `/docs`
- Review mock data in `mock_data/omh_mock_data.json`
- Verify token mappings in the data file

