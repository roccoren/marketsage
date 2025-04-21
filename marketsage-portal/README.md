# MarketSage Portal

MarketSage Portal is the web interface for the MarketSage ecosystem, providing data visualization, interactive dashboards, and configuration management for stock analysis and prediction.

## Overview

The MarketSage Portal provides:

- Interactive data visualization dashboards
- Stock data exploration and analysis
- Prediction results display
- User account management
- System configuration interface
- Real-time monitoring of analysis tasks

## Architecture

The Portal is built using a modern web stack:

- **Frontend**: React.js with TypeScript
- **UI Framework**: Material-UI
- **State Management**: Redux Toolkit
- **Data Visualization**: Plotly.js, D3.js
- **API Integration**: Axios, React Query

## Directory Structure

```
marketsage-portal/
├── src/
│   ├── components/    # Reusable UI components
│   ├── layouts/       # Page layouts and templates
│   ├── assets/        # Static assets (images, styles, fonts)
│   └── services/      # API and service integrations
├── tests/             # Unit and integration tests
├── config/            # Configuration files
├── Dockerfile         # Container definition
├── requirements.txt   # Python dependencies (for build tools)
└── README.md          # This file
```

## Getting Started

### Prerequisites

- Node.js 16+
- npm or yarn
- Access to MarketSage API

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

### Configuration

Create a `.env` file in the root directory with the following variables:

```
VITE_API_URL=http://localhost:8000/api
VITE_AUTH_TOKEN_KEY=marketsage_auth_token
VITE_REFRESH_TOKEN_KEY=marketsage_refresh_token
```

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build production version
- `npm run test` - Run tests
- `npm run lint` - Run linting
- `npm run format` - Format code

### Adding New Components

1. Create a new component in the `src/components` directory
2. Export the component from the appropriate index file
3. Import and use the component where needed

### API Integration

All API calls should go through the service layer:

```typescript
// Example service call
import { stockService } from '../services';

// In a component
const fetchData = async () => {
  try {
    const data = await stockService.getStockData('AAPL');
    // Handle data
  } catch (error) {
    // Handle error
  }
};
```

## Docker

Build and run with Docker:

```bash
# Build the image
docker build -t marketsage-portal .

# Run the container
docker run -p 3000:80 marketsage-portal
```

## Integration with Other Components

The Portal integrates with other MarketSage components through:

1. **MarketSage API**: REST API calls for data and functionality
2. **WebSockets**: Real-time updates and notifications

## License

[MIT License](LICENSE)