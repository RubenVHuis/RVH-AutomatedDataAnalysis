import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Dict, Any, Tuple
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class TimeSeriesForecasting:
    """
    Collection of time series forecasting methods.

    Supports:
    - ARIMA: AutoRegressive Integrated Moving Average
    - SARIMAX: Seasonal ARIMA with exogenous variables
    - Prophet: Meta's forecasting tool for time series with multiple seasonality

    All methods return dictionaries with forecasts, metrics, and diagnostic information.
    """

    # ==================== DIAGNOSTIC TESTS ====================

    @staticmethod
    def augmented_dickey_fuller_test(series: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform Augmented Dickey-Fuller test for stationarity.

        Tests the null hypothesis that a unit root is present in the time series.
        If p-value < alpha, the series is stationary.

        Parameters
        ----------
        series : pd.Series
            Time series data to test.
        alpha : float, optional (default=0.05)
            Significance level for hypothesis testing.

        Returns
        -------
        dict
            Results containing test statistic, p-value, and stationarity conclusion.
        """
        # Clean data
        clean_series = series.dropna()

        if len(clean_series) < 3:
            return {"error": "Insufficient data (need at least 3 observations)"}

        # Perform ADF test
        result = adfuller(clean_series, autolag="AIC")

        adf_statistic = result[0]
        p_value = result[1]
        used_lag = result[2]
        n_obs = result[3]
        critical_values = result[4]

        is_stationary = p_value < alpha

        return {
            "test": "Augmented Dickey-Fuller Test",
            "adf_statistic": adf_statistic,
            "p_value": p_value,
            "used_lag": used_lag,
            "n_observations": n_obs,
            "critical_values": critical_values,
            "alpha": alpha,
            "is_stationary": is_stationary,
            "interpretation": f"Series is {'stationary' if is_stationary else 'non-stationary'} at α={alpha}",
        }

    @staticmethod
    def plot_acf_pacf(
        series: pd.Series, lags: Optional[int] = None, figsize: Tuple[int, int] = (12, 5), title: str = "ACF and PACF"
    ) -> None:
        """
        Plot Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).

        Useful for determining ARIMA parameters (p, d, q).

        Parameters
        ----------
        series : pd.Series
            Time series data.
        lags : int or None, optional (default=None)
            Number of lags to plot. If None, uses min(len(series)//2 - 1, 40).
        figsize : tuple, optional (default=(12, 5))
            Figure size (width, height).
        title : str, optional (default='ACF and PACF')
            Plot title.
        """
        clean_series = series.dropna()

        if lags is None:
            lags = min(len(clean_series) // 2 - 1, 40)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        plot_acf(clean_series, lags=lags, ax=axes[0])
        axes[0].set_title("Autocorrelation Function (ACF)")

        plot_pacf(clean_series, lags=lags, ax=axes[1])
        axes[1].set_title("Partial Autocorrelation Function (PACF)")

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    # ==================== ARIMA ====================

    @staticmethod
    def arima_model(
        series: pd.Series,
        order: Tuple[int, int, int] = (1, 1, 1),
        forecast_periods: int = 10,
        alpha: float = 0.05,
        trend: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Fit ARIMA model and generate forecasts.

        ARIMA(p, d, q):
        - p: number of autoregressive (AR) terms
        - d: number of differences (I)
        - q: number of moving average (MA) terms

        Parameters
        ----------
        series : pd.Series
            Time series data (must be numeric and ideally datetime-indexed).
        order : tuple, optional (default=(1, 1, 1))
            ARIMA order (p, d, q).
        forecast_periods : int, optional (default=10)
            Number of periods to forecast into the future.
        alpha : float, optional (default=0.05)
            Significance level for confidence intervals (e.g., 0.05 for 95% CI).
        trend : str or None, optional (default=None)
            Trend parameter: 'n' (no trend), 'c' (constant), 't' (linear), 'ct' (constant+linear).
        **kwargs
            Additional parameters to pass to ARIMA model.

        Returns
        -------
        dict
            Results containing:
            - model: fitted ARIMA model
            - summary: model summary
            - forecast: forecasted values
            - forecast_ci: confidence intervals for forecasts
            - aic: Akaike Information Criterion
            - bic: Bayesian Information Criterion
            - residuals: model residuals
        """
        # Clean data
        clean_series = series.dropna()

        if len(clean_series) < 10:
            return {"error": "Insufficient data (need at least 10 observations for ARIMA)"}

        try:
            # Fit ARIMA model
            model = ARIMA(clean_series, order=order, trend=trend, **kwargs)
            fitted_model = model.fit()

            # Generate forecast
            forecast_result = fitted_model.get_forecast(steps=forecast_periods, alpha=alpha)
            forecast = forecast_result.predicted_mean
            forecast_ci = forecast_result.conf_int()

            # Extract metrics
            aic = fitted_model.aic
            bic = fitted_model.bic
            residuals = fitted_model.resid

            return {
                "model_type": "ARIMA",
                "order": order,
                "model": fitted_model,
                "summary": fitted_model.summary(),
                "forecast": forecast,
                "forecast_ci": forecast_ci,
                "aic": aic,
                "bic": bic,
                "residuals": residuals,
                "n_observations": len(clean_series),
                "forecast_periods": forecast_periods,
            }

        except Exception as e:
            return {"error": f"ARIMA model failed: {str(e)}"}

    @staticmethod
    def sarimax_model(
        series: pd.Series,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        exog: Optional[pd.DataFrame] = None,
        forecast_periods: int = 10,
        exog_forecast: Optional[pd.DataFrame] = None,
        alpha: float = 0.05,
        trend: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Fit SARIMAX model (Seasonal ARIMA with exogenous variables) and generate forecasts.

        SARIMAX(p, d, q)(P, D, Q, s):
        - (p, d, q): Non-seasonal ARIMA order
        - (P, D, Q, s): Seasonal order (P, D, Q are seasonal AR, I, MA; s is seasonal period)

        Parameters
        ----------
        series : pd.Series
            Time series data (must be numeric).
        order : tuple, optional (default=(1, 1, 1))
            ARIMA order (p, d, q).
        seasonal_order : tuple, optional (default=(0, 0, 0, 0))
            Seasonal ARIMA order (P, D, Q, s).
            Example: (1, 1, 1, 12) for monthly data with yearly seasonality.
        exog : pd.DataFrame or None, optional (default=None)
            Exogenous variables (must have same length as series).
        forecast_periods : int, optional (default=10)
            Number of periods to forecast.
        exog_forecast : pd.DataFrame or None, optional (default=None)
            Future values of exogenous variables for forecasting.
            Required if exog is provided.
        alpha : float, optional (default=0.05)
            Significance level for confidence intervals.
        trend : str or None, optional (default=None)
            Trend parameter: 'n', 'c', 't', 'ct'.
        **kwargs
            Additional parameters to pass to SARIMAX model.

        Returns
        -------
        dict
            Results containing model, forecasts, metrics, and diagnostics.
        """
        # Clean data
        clean_series = series.dropna()

        if len(clean_series) < 10:
            return {"error": "Insufficient data (need at least 10 observations for SARIMAX)"}

        # Validate exogenous variables
        if exog is not None and exog_forecast is None and forecast_periods > 0:
            return {"error": "exog_forecast required when exog is provided and forecast_periods > 0"}

        try:
            # Fit SARIMAX model
            model = SARIMAX(clean_series, exog=exog, order=order, seasonal_order=seasonal_order, trend=trend, **kwargs)
            fitted_model = model.fit(disp=False)

            # Generate forecast
            if exog_forecast is not None:
                forecast_result = fitted_model.get_forecast(steps=forecast_periods, exog=exog_forecast, alpha=alpha)
            else:
                forecast_result = fitted_model.get_forecast(steps=forecast_periods, alpha=alpha)

            forecast = forecast_result.predicted_mean
            forecast_ci = forecast_result.conf_int()

            # Extract metrics
            aic = fitted_model.aic
            bic = fitted_model.bic
            residuals = fitted_model.resid

            return {
                "model_type": "SARIMAX",
                "order": order,
                "seasonal_order": seasonal_order,
                "model": fitted_model,
                "summary": fitted_model.summary(),
                "forecast": forecast,
                "forecast_ci": forecast_ci,
                "aic": aic,
                "bic": bic,
                "residuals": residuals,
                "n_observations": len(clean_series),
                "forecast_periods": forecast_periods,
            }

        except Exception as e:
            return {"error": f"SARIMAX model failed: {str(e)}"}

    @staticmethod
    def auto_arima(
        series: pd.Series,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        seasonal: bool = False,
        m: int = 1,
        forecast_periods: int = 10,
        alpha: float = 0.05,
        information_criterion: str = "aic",
    ) -> Dict[str, Any]:
        """
        Automatically select best ARIMA parameters using grid search.

        Searches over parameter ranges and selects the model with the lowest
        information criterion (AIC or BIC).

        Parameters
        ----------
        series : pd.Series
            Time series data.
        max_p : int, optional (default=5)
            Maximum p parameter to test.
        max_d : int, optional (default=2)
            Maximum d parameter to test.
        max_q : int, optional (default=5)
            Maximum q parameter to test.
        seasonal : bool, optional (default=False)
            Whether to include seasonal components.
        m : int, optional (default=1)
            Seasonal period (e.g., 12 for monthly data with yearly seasonality).
        forecast_periods : int, optional (default=10)
            Number of periods to forecast.
        alpha : float, optional (default=0.05)
            Significance level for confidence intervals.
        information_criterion : str, optional (default='aic')
            Criterion to minimize: 'aic' or 'bic'.

        Returns
        -------
        dict
            Results containing best model, parameters, and forecasts.
        """
        clean_series = series.dropna()

        if len(clean_series) < 20:
            return {"error": "Insufficient data (need at least 20 observations for auto_arima)"}

        best_ic = np.inf
        best_order = None
        best_seasonal_order = None
        best_model = None
        tested_models = []

        # Grid search
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        if seasonal:
                            # Try seasonal ARIMA
                            for P in range(3):
                                for D in range(2):
                                    for Q in range(3):
                                        try:
                                            model = SARIMAX(clean_series, order=(p, d, q), seasonal_order=(P, D, Q, m))
                                            fitted = model.fit(disp=False)

                                            ic = fitted.aic if information_criterion == "aic" else fitted.bic

                                            tested_models.append(
                                                {
                                                    "order": (p, d, q),
                                                    "seasonal_order": (P, D, Q, m),
                                                    information_criterion: ic,
                                                }
                                            )

                                            if ic < best_ic:
                                                best_ic = ic
                                                best_order = (p, d, q)
                                                best_seasonal_order = (P, D, Q, m)
                                                best_model = fitted

                                        except:
                                            continue
                        else:
                            # Non-seasonal ARIMA
                            model = ARIMA(clean_series, order=(p, d, q))
                            fitted = model.fit()

                            ic = fitted.aic if information_criterion == "aic" else fitted.bic

                            tested_models.append({"order": (p, d, q), information_criterion: ic})

                            if ic < best_ic:
                                best_ic = ic
                                best_order = (p, d, q)
                                best_model = fitted

                    except:
                        continue

        if best_model is None:
            return {"error": "No valid ARIMA model found"}

        # Generate forecast with best model
        forecast_result = best_model.get_forecast(steps=forecast_periods, alpha=alpha)
        forecast = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int()

        return {
            "model_type": "Auto ARIMA",
            "best_order": best_order,
            "best_seasonal_order": best_seasonal_order if seasonal else None,
            "model": best_model,
            "summary": best_model.summary(),
            "forecast": forecast,
            "forecast_ci": forecast_ci,
            information_criterion: best_ic,
            "residuals": best_model.resid,
            "tested_models": pd.DataFrame(tested_models),
        }

    # ==================== PROPHET ====================

    @staticmethod
    def prophet_model(
        df: pd.DataFrame,
        date_col: str = "ds",
        value_col: str = "y",
        forecast_periods: int = 10,
        freq: str = "D",
        yearly_seasonality: Union[bool, str, int] = "auto",
        weekly_seasonality: Union[bool, str, int] = "auto",
        daily_seasonality: Union[bool, str, int] = "auto",
        seasonality_mode: str = "additive",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Fit Prophet model and generate forecasts.

        Prophet is designed for time series with strong seasonal effects and several
        seasons of historical data. It works best with daily data with at least one year
        of history.

        Parameters
        ----------
        df : pd.DataFrame
            Time series data. Must contain columns for date and value.
        date_col : str, optional (default='ds')
            Name of the date column. If not 'ds', will be renamed internally.
        value_col : str, optional (default='y')
            Name of the value column. If not 'y', will be renamed internally.
        forecast_periods : int, optional (default=10)
            Number of periods to forecast.
        freq : str, optional (default='D')
            Frequency of the time series: 'D' (daily), 'W' (weekly), 'M' (monthly), etc.
        yearly_seasonality : bool, str, or int, optional (default='auto')
            Fit yearly seasonality. Can be 'auto', True, False, or Fourier order integer.
        weekly_seasonality : bool, str, or int, optional (default='auto')
            Fit weekly seasonality.
        daily_seasonality : bool, str, or int, optional (default='auto')
            Fit daily seasonality.
        seasonality_mode : str, optional (default='additive')
            'additive' or 'multiplicative' seasonality.
        changepoint_prior_scale : float, optional (default=0.05)
            Flexibility of trend changes (higher = more flexible).
        seasonality_prior_scale : float, optional (default=10.0)
            Strength of seasonality (higher = stronger).
        holidays : pd.DataFrame or None, optional (default=None)
            DataFrame with 'holiday' and 'ds' columns for custom holidays.
        **kwargs
            Additional parameters to pass to Prophet.

        Returns
        -------
        dict
            Results containing model, forecast, and components.
        """
        if not PROPHET_AVAILABLE:
            return {
                "error": "Prophet not installed. Install with: pip install prophet",
                "note": "Prophet requires pystan. On some systems, you may need: pip install pystan prophet",
            }

        # Prepare data
        df_prophet = df.copy()

        # Rename columns to Prophet's expected format
        if date_col != "ds":
            df_prophet = df_prophet.rename(columns={date_col: "ds"})
        if value_col != "y":
            df_prophet = df_prophet.rename(columns={value_col: "y"})

        # Keep only required columns
        df_prophet = df_prophet[["ds", "y"]].dropna()

        if len(df_prophet) < 2:
            return {"error": "Insufficient data (need at least 2 observations)"}

        try:
            # Initialize and fit Prophet model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = Prophet(
                    yearly_seasonality=yearly_seasonality,
                    weekly_seasonality=weekly_seasonality,
                    daily_seasonality=daily_seasonality,
                    seasonality_mode=seasonality_mode,
                    changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_prior_scale=seasonality_prior_scale,
                    holidays=holidays,
                    **kwargs,
                )

                model.fit(df_prophet)

            # Create future dataframe
            future = model.make_future_dataframe(periods=forecast_periods, freq=freq)

            # Generate forecast
            forecast = model.predict(future)

            # Extract components
            components = {
                "trend": forecast[["ds", "trend"]],
                "forecast": forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
            }

            # Add seasonal components if they exist
            if "yearly" in forecast.columns:
                components["yearly"] = forecast[["ds", "yearly"]]
            if "weekly" in forecast.columns:
                components["weekly"] = forecast[["ds", "weekly"]]
            if "daily" in forecast.columns:
                components["daily"] = forecast[["ds", "daily"]]

            return {
                "model_type": "Prophet",
                "model": model,
                "forecast": forecast,
                "forecast_future_only": forecast.tail(forecast_periods),
                "components": components,
                "n_observations": len(df_prophet),
                "forecast_periods": forecast_periods,
                "seasonality_mode": seasonality_mode,
            }

        except Exception as e:
            return {"error": f"Prophet model failed: {str(e)}"}

    # ==================== VISUALIZATION ====================

    @staticmethod
    def plot_forecast(
        series: pd.Series,
        forecast: pd.Series,
        forecast_ci: Optional[pd.DataFrame] = None,
        title: str = "Time Series Forecast",
        figsize: Tuple[int, int] = (12, 6),
        xlabel: str = "Time",
        ylabel: str = "Value",
    ) -> None:
        """
        Plot historical data with forecast and confidence intervals.

        Parameters
        ----------
        series : pd.Series
            Historical time series data.
        forecast : pd.Series
            Forecasted values.
        forecast_ci : pd.DataFrame or None, optional (default=None)
            Confidence intervals for forecast (must have 2 columns: lower and upper).
        title : str, optional (default='Time Series Forecast')
            Plot title.
        figsize : tuple, optional (default=(12, 6))
            Figure size (width, height).
        xlabel : str, optional (default='Time')
            X-axis label.
        ylabel : str, optional (default='Value')
            Y-axis label.
        """
        plt.figure(figsize=figsize)

        # Plot historical data
        plt.plot(series.index, series.values, label="Historical", color="blue")

        # Plot forecast
        plt.plot(forecast.index, forecast.values, label="Forecast", color="red", linestyle="--")

        # Plot confidence interval
        if forecast_ci is not None:
            plt.fill_between(
                forecast.index,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1],
                alpha=0.3,
                color="red",
                label="95% Confidence Interval",
            )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_prophet_forecast(forecast: pd.DataFrame, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot Prophet forecast results.

        Parameters
        ----------
        forecast : pd.DataFrame
            Forecast dataframe from Prophet model.
        figsize : tuple, optional (default=(12, 6))
            Figure size (width, height).
        """
        plt.figure(figsize=figsize)

        # Plot forecast
        plt.plot(forecast["ds"], forecast["yhat"], label="Forecast", color="blue")

        # Plot confidence interval
        plt.fill_between(
            forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.3, color="blue", label="95% CI"
        )

        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title("Prophet Forecast")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ==================== EVALUATION METRICS ====================

    @staticmethod
    def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """
        Calculate forecast evaluation metrics.

        Parameters
        ----------
        y_true : pd.Series
            Actual values.
        y_pred : pd.Series
            Predicted values.

        Returns
        -------
        dict
            Dictionary containing MAE, MSE, RMSE, MAPE, and sMAPE.
        """
        # Align series
        df_temp = pd.DataFrame({"true": y_true, "pred": y_pred}).dropna()

        if len(df_temp) == 0:
            return {"error": "No overlapping data points"}

        y_true_clean = df_temp["true"]
        y_pred_clean = df_temp["pred"]

        # Calculate metrics
        mae = np.mean(np.abs(y_true_clean - y_pred_clean))
        mse = np.mean((y_true_clean - y_pred_clean) ** 2)
        rmse = np.sqrt(mse)

        # MAPE (avoid division by zero)
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100 if (y_true_clean != 0).all() else np.nan

        # sMAPE (symmetric MAPE)
        smape = np.mean(2.0 * np.abs(y_true_clean - y_pred_clean) / (np.abs(y_true_clean) + np.abs(y_pred_clean))) * 100

        return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "sMAPE": smape, "n_observations": len(df_temp)}
