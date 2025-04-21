import { configureStore } from '@reduxjs/toolkit';
import { setupListeners } from '@reduxjs/toolkit/query';

// Reducers
import authReducer from './slices/authSlice';
import uiReducer from './slices/uiSlice';
import stocksReducer from './slices/stocksSlice';
import predictionsReducer from './slices/predictionsSlice';

// RTK Query APIs
import { stocksApi } from './api/stocksApi';
import { predictionsApi } from './api/predictionsApi';
import { authApi } from './api/authApi';

export const store = configureStore({
  reducer: {
    auth: authReducer,
    ui: uiReducer,
    stocks: stocksReducer,
    predictions: predictionsReducer,
    [stocksApi.reducerPath]: stocksApi.reducer,
    [predictionsApi.reducerPath]: predictionsApi.reducer,
    [authApi.reducerPath]: authApi.reducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware().concat(
      stocksApi.middleware,
      predictionsApi.middleware,
      authApi.middleware
    ),
  devTools: process.env.NODE_ENV !== 'production',
});

// Optional, but required for refetchOnFocus/refetchOnReconnect behaviors
setupListeners(store.dispatch);

// Export types
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;