import { useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Box, CircularProgress } from '@mui/material';
import { useSelector } from 'react-redux';

// Layouts
import MainLayout from './layouts/MainLayout';
import AuthLayout from './layouts/AuthLayout';

// Pages
import Dashboard from './pages/Dashboard';
import StockExplorer from './pages/StockExplorer';
import StockDetail from './pages/StockDetail';
import Predictions from './pages/Predictions';
import PredictionDetail from './pages/PredictionDetail';
import Settings from './pages/Settings';
import Login from './pages/Login';
import Register from './pages/Register';
import ForgotPassword from './pages/ForgotPassword';
import NotFound from './pages/NotFound';

// Auth
import ProtectedRoute from './components/auth/ProtectedRoute';
import { RootState } from './store';
import { useInitApp } from './hooks/useInitApp';

const App = () => {
  const { isInitializing } = useInitApp();
  const { isAuthenticated, isLoading } = useSelector((state: RootState) => state.auth);

  // Show loading spinner while initializing the app
  if (isInitializing || isLoading) {
    return (
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100vh',
        }}
      >
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Routes>
      {/* Auth routes */}
      <Route element={<AuthLayout />}>
        <Route
          path="/login"
          element={isAuthenticated ? <Navigate to="/dashboard" replace /> : <Login />}
        />
        <Route
          path="/register"
          element={isAuthenticated ? <Navigate to="/dashboard" replace /> : <Register />}
        />
        <Route path="/forgot-password" element={<ForgotPassword />} />
      </Route>

      {/* Protected routes */}
      <Route
        element={
          <ProtectedRoute isAuthenticated={isAuthenticated}>
            <MainLayout />
          </ProtectedRoute>
        }
      >
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/stocks" element={<StockExplorer />} />
        <Route path="/stocks/:symbol" element={<StockDetail />} />
        <Route path="/predictions" element={<Predictions />} />
        <Route path="/predictions/:id" element={<PredictionDetail />} />
        <Route path="/settings" element={<Settings />} />
      </Route>

      {/* 404 Not Found */}
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
};

export default App;