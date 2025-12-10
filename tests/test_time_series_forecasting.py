"""
Unit tests for TimeSeriesForecasting class.
"""

import pytest
import pandas as pd
import numpy as np
from src.time_series_forecasting import TimeSeriesForecasting, PROPHET_AVAILABLE
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper


class TestTimeSeriesForecasting:
    """Test suite for TimeSeriesForecasting class."""

    @pytest.fixture
    def simple_time_series(self):
        """Create simple time series data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        # Trend + noise
        values = np.arange(100) * 0.5 + np.random.randn(100) * 2
        series = pd.Series(values, index=dates, name='value')
        return series

    @pytest.fixture
    def seasonal_time_series(self):
        """Create time series with seasonal pattern."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=365*2, freq='D')
        # Trend + yearly seasonality + noise
        t = np.arange(len(dates))
        trend = t * 0.1
        seasonal = 10 * np.sin(2 * np.pi * t / 365)
        noise = np.random.randn(len(dates)) * 2
        values = trend + seasonal + noise + 50
        series = pd.Series(values, index=dates, name='value')
        return series

    @pytest.fixture
    def stationary_series(self):
        """Create stationary time series."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        values = np.random.randn(100)
        series = pd.Series(values, index=dates, name='value')
        return series

    @pytest.fixture
    def prophet_dataframe(self):
        """Create dataframe in Prophet format."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=365*2, freq='D')
        t = np.arange(len(dates))
        trend = t * 0.1
        seasonal = 10 * np.sin(2 * np.pi * t / 365)
        values = trend + seasonal + np.random.randn(len(dates)) * 2 + 50

        df = pd.DataFrame({
            'ds': dates,
            'y': values
        })
        return df

    # ==================== DIAGNOSTIC TESTS ====================

    def test_augmented_dickey_fuller_test_stationary(self, stationary_series):
        """Test ADF test on stationary series."""
        result = TimeSeriesForecasting.augmented_dickey_fuller_test(stationary_series)

        assert 'test' in result
        assert 'adf_statistic' in result
        assert 'p_value' in result
        assert 'is_stationary' in result
        assert 'interpretation' in result
        assert isinstance(result['p_value'], float)
        assert 0 <= result['p_value'] <= 1
        assert result['is_stationary'] == True  # Should be stationary

    def test_augmented_dickey_fuller_test_non_stationary(self, simple_time_series):
        """Test ADF test on non-stationary series."""
        result = TimeSeriesForecasting.augmented_dickey_fuller_test(simple_time_series)

        assert 'adf_statistic' in result
        assert 'p_value' in result
        assert 'critical_values' in result
        assert isinstance(result['critical_values'], dict)

    def test_augmented_dickey_fuller_test_insufficient_data(self):
        """Test ADF test with insufficient data."""
        series = pd.Series([1, 2])
        result = TimeSeriesForecasting.augmented_dickey_fuller_test(series)

        assert 'error' in result

    # ==================== ARIMA TESTS ====================

    def test_arima_model_basic(self, simple_time_series):
        """Test basic ARIMA model fitting and forecasting."""
        result = TimeSeriesForecasting.arima_model(
            series=simple_time_series,
            order=(1, 1, 1),
            forecast_periods=10
        )

        assert 'model_type' in result
        assert result['model_type'] == 'ARIMA'
        assert 'order' in result
        assert result['order'] == (1, 1, 1)
        assert 'model' in result
        assert 'forecast' in result
        assert 'forecast_ci' in result
        assert 'aic' in result
        assert 'bic' in result
        assert 'residuals' in result

        # Check forecast length
        assert len(result['forecast']) == 10
        assert result['forecast_ci'].shape == (10, 2)

        # Check AIC/BIC are finite numbers
        assert np.isfinite(result['aic'])
        assert np.isfinite(result['bic'])

    def test_arima_model_different_orders(self, simple_time_series):
        """Test ARIMA with different order parameters."""
        orders = [(0, 1, 0), (1, 0, 0), (2, 1, 2)]

        for order in orders:
            result = TimeSeriesForecasting.arima_model(
                series=simple_time_series,
                order=order,
                forecast_periods=5
            )

            assert result['order'] == order
            assert len(result['forecast']) == 5

    def test_arima_model_with_trend(self, simple_time_series):
        """Test ARIMA with trend parameter."""
        result = TimeSeriesForecasting.arima_model(
            series=simple_time_series,
            order=(1, 1, 1),
            forecast_periods=5,
            trend='t'
        )

        assert 'model' in result
        assert len(result['forecast']) == 5

    def test_arima_model_insufficient_data(self):
        """Test ARIMA with insufficient data."""
        series = pd.Series([1, 2, 3, 4, 5])
        result = TimeSeriesForecasting.arima_model(series, order=(1, 1, 1))

        assert 'error' in result

    # ==================== SARIMAX TESTS ====================

    def test_sarimax_model_non_seasonal(self, simple_time_series):
        """Test SARIMAX without seasonal component."""
        result = TimeSeriesForecasting.sarimax_model(
            series=simple_time_series,
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
            forecast_periods=10
        )

        assert 'model_type' in result
        assert result['model_type'] == 'SARIMAX'
        assert 'order' in result
        assert 'seasonal_order' in result
        assert 'forecast' in result
        assert 'forecast_ci' in result
        assert len(result['forecast']) == 10

    def test_sarimax_model_with_seasonality(self, seasonal_time_series):
        """Test SARIMAX with seasonal component."""
        result = TimeSeriesForecasting.sarimax_model(
            series=seasonal_time_series,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 7),  # Weekly seasonality
            forecast_periods=14
        )

        assert result['seasonal_order'] == (1, 0, 1, 7)
        assert len(result['forecast']) == 14
        assert 'aic' in result
        assert 'bic' in result

    def test_sarimax_model_with_exogenous(self, simple_time_series):
        """Test SARIMAX with exogenous variables."""
        np.random.seed(42)
        exog = pd.DataFrame({
            'exog1': np.random.randn(len(simple_time_series)),
            'exog2': np.random.randn(len(simple_time_series))
        }, index=simple_time_series.index)

        # Create future exogenous variables
        exog_forecast = pd.DataFrame({
            'exog1': np.random.randn(10),
            'exog2': np.random.randn(10)
        })

        result = TimeSeriesForecasting.sarimax_model(
            series=simple_time_series,
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
            exog=exog,
            exog_forecast=exog_forecast,
            forecast_periods=10
        )

        assert 'forecast' in result
        assert len(result['forecast']) == 10

    def test_sarimax_model_exog_missing_forecast(self, simple_time_series):
        """Test SARIMAX with exogenous but missing forecast values."""
        exog = pd.DataFrame({
            'exog1': np.random.randn(len(simple_time_series))
        }, index=simple_time_series.index)

        result = TimeSeriesForecasting.sarimax_model(
            series=simple_time_series,
            order=(1, 1, 1),
            exog=exog,
            forecast_periods=10
        )

        assert 'error' in result

    def test_sarimax_model_insufficient_data(self):
        """Test SARIMAX with insufficient data."""
        series = pd.Series([1, 2, 3])
        result = TimeSeriesForecasting.sarimax_model(series, order=(1, 1, 1))

        assert 'error' in result

    # ==================== AUTO ARIMA TESTS ====================

    def test_auto_arima_non_seasonal(self, simple_time_series):
        """Test auto ARIMA parameter selection."""
        result = TimeSeriesForecasting.auto_arima(
            series=simple_time_series,
            max_p=2,
            max_d=1,
            max_q=2,
            seasonal=False,
            forecast_periods=5
        )

        assert 'model_type' in result
        assert result['model_type'] == 'Auto ARIMA'
        assert 'best_order' in result
        assert 'forecast' in result
        assert 'tested_models' in result
        assert len(result['forecast']) == 5

        # Check best order is within bounds
        p, d, q = result['best_order']
        assert 0 <= p <= 2
        assert 0 <= d <= 1
        assert 0 <= q <= 2

        # Check tested models dataframe
        assert isinstance(result['tested_models'], pd.DataFrame)
        assert 'order' in result['tested_models'].columns

    def test_auto_arima_with_seasonal(self, seasonal_time_series):
        """Test auto ARIMA with seasonal component."""
        result = TimeSeriesForecasting.auto_arima(
            series=seasonal_time_series,
            max_p=1,
            max_d=1,
            max_q=1,
            seasonal=True,
            m=7,
            forecast_periods=7
        )

        assert 'best_order' in result
        assert 'best_seasonal_order' in result
        assert result['best_seasonal_order'] is not None
        assert len(result['forecast']) == 7

    def test_auto_arima_information_criterion(self, simple_time_series):
        """Test auto ARIMA with different information criteria."""
        result_aic = TimeSeriesForecasting.auto_arima(
            series=simple_time_series,
            max_p=2,
            max_d=1,
            max_q=2,
            information_criterion='aic',
            forecast_periods=5
        )

        result_bic = TimeSeriesForecasting.auto_arima(
            series=simple_time_series,
            max_p=2,
            max_d=1,
            max_q=2,
            information_criterion='bic',
            forecast_periods=5
        )

        assert 'aic' in result_aic
        assert 'bic' in result_bic
        # BIC typically selects simpler models
        # Both should have valid forecasts
        assert len(result_aic['forecast']) == 5
        assert len(result_bic['forecast']) == 5

    def test_auto_arima_insufficient_data(self):
        """Test auto ARIMA with insufficient data."""
        series = pd.Series(range(15))
        result = TimeSeriesForecasting.auto_arima(series, max_p=2, max_d=1, max_q=2)

        assert 'error' in result

    # ==================== PROPHET TESTS ====================

    @pytest.mark.skipif(not PROPHET_AVAILABLE, reason="Prophet not installed")
    def test_prophet_model_basic(self, prophet_dataframe):
        """Test basic Prophet model fitting and forecasting."""
        result = TimeSeriesForecasting.prophet_model(
            df=prophet_dataframe,
            forecast_periods=30,
            freq='D'
        )

        assert 'model_type' in result
        assert result['model_type'] == 'Prophet'
        assert 'model' in result
        assert 'forecast' in result
        assert 'forecast_future_only' in result
        assert 'components' in result

        # Check forecast length (includes historical + future)
        assert len(result['forecast']) == len(prophet_dataframe) + 30
        assert len(result['forecast_future_only']) == 30

        # Check required forecast columns
        forecast_cols = ['yhat', 'yhat_lower', 'yhat_upper', 'ds']
        for col in forecast_cols:
            assert col in result['forecast'].columns

    @pytest.mark.skipif(not PROPHET_AVAILABLE, reason="Prophet not installed")
    def test_prophet_model_custom_columns(self, prophet_dataframe):
        """Test Prophet with custom column names."""
        df_custom = prophet_dataframe.copy()
        df_custom = df_custom.rename(columns={'ds': 'date', 'y': 'value'})

        result = TimeSeriesForecasting.prophet_model(
            df=df_custom,
            date_col='date',
            value_col='value',
            forecast_periods=10
        )

        assert 'forecast' in result
        assert len(result['forecast_future_only']) == 10

    @pytest.mark.skipif(not PROPHET_AVAILABLE, reason="Prophet not installed")
    def test_prophet_model_seasonality_modes(self, prophet_dataframe):
        """Test Prophet with different seasonality modes."""
        for mode in ['additive', 'multiplicative']:
            result = TimeSeriesForecasting.prophet_model(
                df=prophet_dataframe,
                forecast_periods=10,
                seasonality_mode=mode
            )

            assert 'seasonality_mode' in result
            assert result['seasonality_mode'] == mode

    @pytest.mark.skipif(not PROPHET_AVAILABLE, reason="Prophet not installed")
    def test_prophet_model_custom_seasonality(self, prophet_dataframe):
        """Test Prophet with custom seasonality settings."""
        result = TimeSeriesForecasting.prophet_model(
            df=prophet_dataframe,
            forecast_periods=10,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )

        assert 'forecast' in result
        assert len(result['forecast_future_only']) == 10

    @pytest.mark.skipif(not PROPHET_AVAILABLE, reason="Prophet not installed")
    def test_prophet_model_with_holidays(self, prophet_dataframe):
        """Test Prophet with custom holidays."""
        holidays = pd.DataFrame({
            'holiday': ['new_year', 'christmas'],
            'ds': pd.to_datetime(['2020-01-01', '2020-12-25'])
        })

        result = TimeSeriesForecasting.prophet_model(
            df=prophet_dataframe,
            forecast_periods=10,
            holidays=holidays
        )

        assert 'forecast' in result

    @pytest.mark.skipif(not PROPHET_AVAILABLE, reason="Prophet not installed")
    def test_prophet_model_insufficient_data(self):
        """Test Prophet with insufficient data."""
        df = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=1),
            'y': [10]
        })

        result = TimeSeriesForecasting.prophet_model(df, forecast_periods=5)

        assert 'error' in result

    def test_prophet_model_not_installed(self, prophet_dataframe, monkeypatch):
        """Test Prophet error when not installed."""
        # Temporarily set PROPHET_AVAILABLE to False
        import src.time_series_forecasting as tsf_module
        monkeypatch.setattr(tsf_module, 'PROPHET_AVAILABLE', False)

        result = TimeSeriesForecasting.prophet_model(
            df=prophet_dataframe,
            forecast_periods=10
        )

        assert 'error' in result
        assert 'Prophet not installed' in result['error']

    # ==================== EVALUATION TESTS ====================

    def test_evaluate_forecast(self):
        """Test forecast evaluation metrics."""
        np.random.seed(42)
        y_true = pd.Series(np.arange(20) + np.random.randn(20) * 0.5)
        y_pred = pd.Series(np.arange(20) + np.random.randn(20) * 0.5)

        result = TimeSeriesForecasting.evaluate_forecast(y_true, y_pred)

        assert 'MAE' in result
        assert 'MSE' in result
        assert 'RMSE' in result
        assert 'MAPE' in result
        assert 'sMAPE' in result
        assert 'n_observations' in result

        # Check metric properties
        assert result['MAE'] >= 0
        assert result['MSE'] >= 0
        assert result['RMSE'] >= 0
        assert result['RMSE'] == np.sqrt(result['MSE'])
        assert 0 <= result['sMAPE'] <= 200  # sMAPE range is 0-200%

    def test_evaluate_forecast_perfect_prediction(self):
        """Test evaluation with perfect predictions."""
        y_true = pd.Series(range(20))
        y_pred = pd.Series(range(20))

        result = TimeSeriesForecasting.evaluate_forecast(y_true, y_pred)

        assert result['MAE'] == 0
        assert result['MSE'] == 0
        assert result['RMSE'] == 0
        assert result['sMAPE'] == 0

    def test_evaluate_forecast_with_zeros(self):
        """Test evaluation when true values contain zeros."""
        y_true = pd.Series([0, 1, 2, 3, 0, 5])
        y_pred = pd.Series([0.1, 1.1, 2.1, 3.1, 0.1, 5.1])

        result = TimeSeriesForecasting.evaluate_forecast(y_true, y_pred)

        # MAPE should be NaN due to zeros in y_true
        assert np.isnan(result['MAPE'])
        # But other metrics should work
        assert result['MAE'] >= 0
        assert result['RMSE'] >= 0

    def test_evaluate_forecast_no_overlap(self):
        """Test evaluation with no overlapping data."""
        y_true = pd.Series([1, 2, 3], index=[0, 1, 2])
        y_pred = pd.Series([4, 5, 6], index=[3, 4, 5])

        result = TimeSeriesForecasting.evaluate_forecast(y_true, y_pred)

        assert 'error' in result

    # ==================== DETERMINISM TESTS ====================

    def test_arima_determinism(self, simple_time_series):
        """Test that ARIMA produces identical results."""
        result1 = TimeSeriesForecasting.arima_model(
            series=simple_time_series,
            order=(1, 1, 1),
            forecast_periods=10
        )

        result2 = TimeSeriesForecasting.arima_model(
            series=simple_time_series,
            order=(1, 1, 1),
            forecast_periods=10
        )

        # Forecasts should be identical
        assert np.allclose(result1['forecast'].values, result2['forecast'].values)
        assert np.allclose(result1['aic'], result2['aic'])

    # ==================== EDGE CASE TESTS ====================

    def test_arima_with_missing_values(self):
        """Test ARIMA handling of missing values."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        values = np.arange(100) * 0.5 + np.random.randn(100) * 2
        values[10:15] = np.nan  # Introduce missing values
        series = pd.Series(values, index=dates)

        result = TimeSeriesForecasting.arima_model(
            series=series,
            order=(1, 1, 1),
            forecast_periods=10
        )

        # Should handle missing values by dropping them
        assert 'forecast' in result
        assert len(result['forecast']) == 10

    def test_forecast_with_very_short_horizon(self, simple_time_series):
        """Test forecasting with very short horizon."""
        result = TimeSeriesForecasting.arima_model(
            series=simple_time_series,
            order=(1, 1, 1),
            forecast_periods=1
        )

        assert len(result['forecast']) == 1

    def test_forecast_with_long_horizon(self, simple_time_series):
        """Test forecasting with longer horizon."""
        result = TimeSeriesForecasting.arima_model(
            series=simple_time_series,
            order=(1, 1, 1),
            forecast_periods=50
        )

        assert len(result['forecast']) == 50
        # Confidence intervals should widen with longer horizon
        ci_width_start = result['forecast_ci'].iloc[0, 1] - result['forecast_ci'].iloc[0, 0]
        ci_width_end = result['forecast_ci'].iloc[-1, 1] - result['forecast_ci'].iloc[-1, 0]
        assert ci_width_end > ci_width_start

    # ==================== CONFIDENCE INTERVAL TESTS ====================

    def test_arima_confidence_intervals(self, simple_time_series):
        """Test ARIMA confidence interval properties."""
        result = TimeSeriesForecasting.arima_model(
            series=simple_time_series,
            order=(1, 1, 1),
            forecast_periods=10,
            alpha=0.05  # 95% CI
        )

        forecast = result['forecast']
        forecast_ci = result['forecast_ci']

        # Forecast should be between lower and upper bounds
        for i in range(len(forecast)):
            assert forecast_ci.iloc[i, 0] <= forecast.iloc[i] <= forecast_ci.iloc[i, 1]

    def test_sarimax_confidence_intervals(self, simple_time_series):
        """Test SARIMAX confidence interval properties."""
        result = TimeSeriesForecasting.sarimax_model(
            series=simple_time_series,
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
            forecast_periods=10,
            alpha=0.1  # 90% CI
        )

        forecast = result['forecast']
        forecast_ci = result['forecast_ci']

        # Forecast should be between bounds
        for i in range(len(forecast)):
            assert forecast_ci.iloc[i, 0] <= forecast.iloc[i] <= forecast_ci.iloc[i, 1]

    # ==================== RESIDUAL TESTS ====================

    def test_arima_residuals(self, simple_time_series):
        """Test ARIMA residual properties."""
        result = TimeSeriesForecasting.arima_model(
            series=simple_time_series,
            order=(1, 1, 1),
            forecast_periods=10
        )

        residuals = result['residuals']

        # Residuals should exist and have reasonable properties
        assert len(residuals) > 0
        # Mean should be close to zero for a good model
        assert abs(residuals.mean()) < 2.0
        # Should have finite variance
        assert np.isfinite(residuals.std())

    # ==================== MULTIPLE SERIES TESTS ====================

    def test_multiple_arima_models(self, simple_time_series):
        """Test fitting multiple ARIMA models sequentially."""
        orders = [(1, 0, 0), (0, 1, 1), (1, 1, 1)]
        results = []

        for order in orders:
            result = TimeSeriesForecasting.arima_model(
                series=simple_time_series,
                order=order,
                forecast_periods=5
            )
            results.append(result)

        # All should succeed
        assert all('forecast' in r for r in results)
        # Different orders should give different AIC values
        aics = [r['aic'] for r in results]
        assert len(set(aics)) > 1
