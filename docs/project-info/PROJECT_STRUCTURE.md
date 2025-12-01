# Project Structure

This document describes the organized structure of the AutoRL ARM Edition hackathon submission.

## Directory Organization

```
autorl-arm-edition-hackathon-submission/
│
├── README.md                    # Main project README
├── HACKATHON_SUBMISSION.md      # Hackathon submission details
├── DEVPOST_SUBMISSION.md        # Devpost submission guide
├── PROJECT_STRUCTURE.md         # This file
│
├── backend/                     # Python backend code
│   ├── agent_service/          # AI agent service implementations
│   ├── agents/                 # Agent registry
│   ├── error_handling/         # Error handling utilities
│   ├── llm/                    # LLM integration
│   ├── perception/             # Visual perception
│   ├── planner/                # Planning logic
│   ├── plugins/                # Plugin system
│   ├── rl/                     # Reinforcement learning
│   ├── runtime/                # Runtime management
│   ├── security/               # Security utilities
│   ├── marketplace/            # Plugin marketplace
│   ├── production_readiness/   # Production utilities
│   ├── examples/               # Example scripts
│   ├── tools/                  # Utility tools
│   ├── autorl_project/         # Legacy project code
│   ├── *.py                    # Main backend entry points
│   └── requirements*.txt       # Python dependencies
│
├── frontend/                    # React frontend
│   ├── components/             # React components
│   ├── pages/                  # Page components
│   ├── contexts/               # React contexts
│   ├── hooks/                  # Custom React hooks
│   ├── services/               # API services
│   ├── lib/                    # Utility libraries
│   ├── assets/                 # Static assets
│   ├── public/                 # Public assets
│   ├── landing-page/           # Landing page
│   ├── *.jsx, *.js, *.css      # Frontend source files
│   └── package.json            # Node.js dependencies
│
├── mobile/                     # Mobile application
│   └── android/                # Android project
│
├── models/                      # ML models
│   └── model/                  # Model export and quantization
│
├── scripts/                     # Build and utility scripts
│   ├── start_autorl_unix.sh   # Unix startup script
│   └── start_autorl_windows.ps1 # Windows startup script
│
├── config/                      # Configuration files
│   ├── config.yaml             # Main configuration
│   ├── env.template             # Environment template
│   ├── prometheus.yml           # Prometheus config
│   └── supabase/               # Supabase config
│
├── docs/                        # Documentation
│   ├── HACKATHON_SUBMISSION.md # Hackathon submission
│   ├── DEVPOST_SUBMISSION.md   # Devpost submission
│   ├── README.md               # Detailed README
│   └── ...                     # Additional documentation
│
├── tests/                       # Test suites
│   └── ...                     # Test files
│
├── demo/                        # Demo scripts and assets
│   ├── run_demo.sh             # Demo script
│   └── test_screen.png         # Test image
│
├── ci/                          # CI/CD configuration
│   └── android-build.yml       # Android build CI
│
├── deployment/                  # Deployment configurations
│   ├── docker-compose.yml      # Docker compose
│   └── prometheus.yml          # Prometheus config
│
├── docker-compose.yml           # Root docker-compose
├── Dockerfile                   # Root Dockerfile
└── .gitignore                   # Git ignore rules
```

## Key Entry Points

### Backend
- `backend/start_autorl.py` - Main startup script
- `backend/servers/master_backend.py` - Master backend server
- `backend/servers/backend_server.py` - Backend server
- `backend/servers/main.py` - Alternative entry point
- `backend/core/orchestrator.py` - Core orchestrator

### Frontend
- `frontend/main.jsx` - React entry point
- `frontend/index.html` - HTML entry point

### Mobile
- `mobile/android/` - Android project root

## Configuration

- Main config: `config/config.yaml`
- Environment: `config/env.template`
- Docker: `docker-compose.yml`

## Documentation

- Main README: `README.md`
- Hackathon submission: `HACKATHON_SUBMISSION.md`
- Detailed docs: `docs/` directory

