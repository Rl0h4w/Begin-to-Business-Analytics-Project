import numpy as np
import numpy_financial as npf
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from tqdm import tqdm
import warnings
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')


class ModelConfig:
    """Configuration class for financial model parameters in USD"""
    def __init__(self):
        # Population and macro
        self.POPULATION = 87.5e6
        self.MARKET_VOLUME_USD = 8.0e9
        self.MARKET_GROWTH = 0.05  # annual growth in USD terms
        self.SEGMENT_SHARE = 0.176
        
        # Inflation in USD
        self.INFLATION_USD = 0.024
        
        # Product parameters
        self.DOSAGE_MG = 2000
        self.PACKS_PER_PATIENT = 24
        self.PRICE_USD = 500   # higher price due to product uniqueness
        self.PRODUCT_SHARE_TARGET = 0.6
        # Ramp up over 8 periods: 0 to 7
        self.PRODUCT_RAMP_UP = np.array([0, 0.2, 0.6, 0.8, 1, 1, 1, 1])
        
        # Patient flow parameters
        self.PREVALENCE = 2160
        self.NEW_CASES_RATE = 0.0000416
        self.DIAGNOSIS_RATE = 0.3
        self.ELIGIBILITY_RATE = 0.75
        self.MORTALITY_RATE = 0.017

        # Financial parameters
        self.VARIABLE_COST_USD = 50
        self.SGA_RATE = 0.35
        self.CAPEX_STARTUP_USD = 2e6
        self.CAPEX_PER_STEP_USD = 2e6
        self.CAPEX_STEP_VOLUME = 50_000

        self.PROMO_COST_PER_PATIENT_USD = 200
        self.WORKING_CAPITAL_RATE = 0.12
        self.TAX_RATE = 0.25
        self.WACC = 0.10
        self.FORECAST_HORIZON = 7
        self.NUM_SIMULATIONS = 10000
        
        # Regions
        self.REGIONS = ['North', 'South', 'East', 'West', 'Central']
        self.REGION_POPULATIONS = np.array([20, 25, 15, 18, 22]) * 1e6
        self.REGION_PREVALENCE = np.array([0.0020, 0.0025, 0.0018, 0.0022, 0.0021])
        self.REGION_PENETRATION = np.array([0.15, 0.12, 0.08, 0.10, 0.11])

        # Competitors (will normalize shares if sum > 1)
        self.COMPETITOR_SHARES = {
            'Competitor 1': pd.Series([0.3, 0.28, 0.25, 0.23, 0.2, 0.18, 0.15, 0.12]),
            'Competitor 2': pd.Series([0.2, 0.19, 0.18, 0.17, 0.15, 0.14, 0.12, 0.1]),
            'Others':       pd.Series([0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15])
        }
        self.COMPETITOR_PRICE_FACTORS = {
            'Competitor 1': 1.1,
            'Competitor 2': 0.9
        }


class FinancialModel:
    """Main financial model class (in USD) with improved logic"""
    def __init__(self, config=None):
        self.config = config or ModelConfig()
        self.years = np.arange(0, self.config.FORECAST_HORIZON + 1)
        
        self.market_data = None
        self.patient_data = None
        self.sales_data = None
        self.financials = None
        self.cash_flows = None
        self.regional_data = None
        self.regional_time_series = None
        self.competitive_data = None

    def run_model(self):
        """Runs all components of the financial model"""
        self._calculate_market_forecast()
        self._calculate_patient_flow()
        self._calculate_sales_forecast()
        self._calculate_financials()
        self._calculate_cash_flows()
        self._calculate_regional_metrics()
        self._calculate_competitive_metrics()

    def _calculate_market_forecast(self):
        """Calculates the forecasted market volume in USD over the horizon"""
        market_outlook_values = [self.config.MARKET_VOLUME_USD]
        for i in range(self.config.FORECAST_HORIZON):
            new_volume = market_outlook_values[-1] * (1 + self.config.MARKET_GROWTH)
            market_outlook_values.append(new_volume)
        self.market_data = pd.Series(market_outlook_values, index=self.years)

    def _calculate_patient_flow(self):
        """Calculates patient flow over the forecast horizon"""
        patients = np.zeros(self.config.FORECAST_HORIZON + 1)
        patients[0] = self.config.PREVALENCE

        for i in range(1, self.config.FORECAST_HORIZON + 1):
            new_cases = self.config.NEW_CASES_RATE * self.config.POPULATION
            diagnosed = new_cases * self.config.DIAGNOSIS_RATE
            eligible = diagnosed * self.config.ELIGIBILITY_RATE
            deaths = patients[i-1] * self.config.MORTALITY_RATE
            patients[i] = max(patients[i-1] + eligible - deaths, 0)
        self.patient_data = pd.Series(patients, index=self.years)

    def _calculate_sales_forecast(self):
        """Calculates sales forecast with USD-based pricing"""
        self.sales_data = pd.DataFrame(index=self.years)

        # Patients on product
        self.sales_data['Patients on Product'] = (
            self.patient_data *
            self.config.PRODUCT_SHARE_TARGET *
            self.config.PRODUCT_RAMP_UP
        )

        self.sales_data['Sales Volume, packs'] = (
            self.sales_data['Patients on Product'] * self.config.PACKS_PER_PATIENT
        )

        usd_inflation_factors = (1 + self.config.INFLATION_USD) ** self.years
        price_usd = self.config.PRICE_USD * usd_inflation_factors
        self.sales_data['Price'] = price_usd
        self.sales_data['Sales'] = self.sales_data['Sales Volume, packs'] * self.sales_data['Price']

        segment_value = self.market_data * self.config.SEGMENT_SHARE
        raw_share = self.sales_data['Sales'] / segment_value
        self.sales_data['Segment Share'] = raw_share.clip(upper=1.0)

        self.sales_data['Revenue Growth'] = self.sales_data['Sales'].pct_change().fillna(0)
        self.sales_data['Volume Growth'] = self.sales_data['Sales Volume, packs'].pct_change().fillna(0)
        self.sales_data['Patient Growth'] = self.sales_data['Patients on Product'].pct_change().fillna(0)

        self.sales_data['Revenue per Patient'] = np.where(
            self.sales_data['Patients on Product'] > 0,
            self.sales_data['Sales'] / self.sales_data['Patients on Product'],
            0
        )

    def _calculate_financials(self):
        """Calculates financial metrics over the forecast horizon in USD"""
        self.financials = pd.DataFrame(index=self.years)
        self.financials['Sales'] = self.sales_data['Sales']

        usd_inflation_factors = (1 + self.config.INFLATION_USD) ** self.years
        cogs_per_pack_usd = self.config.VARIABLE_COST_USD * usd_inflation_factors
        self.financials['COGS'] = self.sales_data['Sales Volume, packs'] * cogs_per_pack_usd

        self.financials['Gross Profit'] = self.financials['Sales'] - self.financials['COGS']
        self.financials['Gross Margin'] = np.where(
            self.financials['Sales'] > 0,
            self.financials['Gross Profit'] / self.financials['Sales'],
            0
        )

        promo_cost_per_patient_usd = self.config.PROMO_COST_PER_PATIENT_USD * usd_inflation_factors
        promo_spend = self.sales_data['Patients on Product'] * promo_cost_per_patient_usd
        base_sg_a = self.financials['Sales'] * self.config.SGA_RATE
        self.financials['SG&A'] = base_sg_a + promo_spend

        self.financials['EBITDA'] = self.financials['Gross Profit'] - self.financials['SG&A']
        self.financials['EBITDA Margin'] = np.where(
            self.financials['Sales'] > 0,
            self.financials['EBITDA'] / self.financials['Sales'],
            0
        )

        # CAPEX: startup + step increments for volume
        expansion_capex = np.zeros_like(self.years, dtype=float)
        expansion_capex[0] = self.config.CAPEX_STARTUP_USD
        volume = self.sales_data['Sales Volume, packs']
        for i, vol in enumerate(volume):
            if i > 0:
                steps = max(0, np.floor((vol - self.config.CAPEX_STEP_VOLUME)/self.config.CAPEX_STEP_VOLUME))
                if steps > 0:
                    expansion_capex[i] = steps * self.config.CAPEX_PER_STEP_USD

        maintenance_capex = self.financials['Sales'] * 0.03
        self.financials['Expansion CAPEX'] = expansion_capex
        self.financials['Additional CAPEX'] = 0
        self.financials['Maintenance CAPEX'] = maintenance_capex
        self.financials['CAPEX'] = (
            self.financials['Expansion CAPEX'] +
            self.financials['Additional CAPEX'] +
            self.financials['Maintenance CAPEX']
        )

        # D&A over 7 years
        horizon = self.config.FORECAST_HORIZON + 1
        da = np.zeros(horizon)
        for i in range(horizon):
            capex_i = self.financials['CAPEX'].iloc[i]
            end_year = min(i + 7, horizon)
            annual_depr = capex_i / 7.0
            for j in range(i, end_year):
                da[j] += annual_depr
        self.financials['D&A'] = da

        self.financials['EBIT'] = self.financials['EBITDA'] - self.financials['D&A']
        self.financials['EBIT Margin'] = np.where(
            self.financials['Sales'] > 0,
            self.financials['EBIT'] / self.financials['Sales'],
            0
        )

        self.financials['EBT'] = self.financials['EBIT']
        self.financials['Tax'] = self.financials['EBT'] * self.config.TAX_RATE
        self.financials['Net Income'] = self.financials['EBT'] - self.financials['Tax']

        net_working_capital = self.financials['Sales'] * self.config.WORKING_CAPITAL_RATE
        self.financials['Change in NWC'] = net_working_capital.diff().fillna(net_working_capital.iloc[0])

        invested_capital = self.financials['CAPEX'].cumsum() + net_working_capital
        self.financials['ROIC'] = np.where(
            invested_capital != 0,
            (self.financials['EBIT'] * (1 - self.config.TAX_RATE)) / invested_capital,
            0
        )

    def _calculate_cash_flows(self):
        """Calculates cash flows based on financial metrics"""
        self.cash_flows = pd.DataFrame(index=self.years)
        self.cash_flows['EBITDA'] = self.financials['EBITDA']
        self.cash_flows['D&A'] = self.financials['D&A']
        self.cash_flows['Tax'] = self.financials['Tax']
        self.cash_flows['Change in NWC'] = self.financials['Change in NWC']

        self.cash_flows['CFO'] = (
            self.financials['EBITDA'] - 
            self.cash_flows['Tax'] - 
            self.cash_flows['Change in NWC'] +
            self.financials['D&A']
        )

        self.cash_flows['CFI'] = -self.financials['CAPEX']
        self.cash_flows['FCF'] = self.cash_flows['CFO'] + self.cash_flows['CFI']

        self.cash_flows['FCF Margin'] = np.where(
            self.financials['Sales'] > 0,
            self.cash_flows['FCF'] / self.financials['Sales'],
            0
        )

        self.cash_flows['Cumulative FCF'] = self.cash_flows['FCF'].cumsum()
        self.cash_flows['Cash Conversion'] = np.where(
            self.financials['EBITDA'] > 0,
            self.cash_flows['CFO'] / self.financials['EBITDA'],
            0
        )

    def _calculate_regional_metrics(self):
        """Calculates regional metrics in USD"""
        self.regional_data = pd.DataFrame(
            index=self.config.REGIONS,
            columns=[
                'Population',
                'Patient Pool',
                'Current Patients',
                'Penetration Rate',
                'Revenue Potential (USD)',
                'Growth Index'
            ]
        )

        self.regional_data['Population'] = self.config.REGION_POPULATIONS
        self.regional_data['Patient Pool'] = (
            self.config.REGION_POPULATIONS * 
            self.config.REGION_PREVALENCE
        )
        self.regional_data['Penetration Rate'] = self.config.REGION_PENETRATION
        self.regional_data['Current Patients'] = (
            self.regional_data['Patient Pool'] * 
            self.regional_data['Penetration Rate']
        )

        base_price_usd = self.config.PRICE_USD
        self.regional_data['Revenue Potential (USD)'] = (
            self.regional_data['Patient Pool'] *
            self.config.PRODUCT_SHARE_TARGET *
            self.config.PACKS_PER_PATIENT *
            base_price_usd
        )

        self.regional_data['Growth Index'] = (
            (1 - self.regional_data['Penetration Rate']) *
            (self.config.REGION_PREVALENCE / self.config.REGION_PREVALENCE.mean())
        )

        self.regional_time_series = {}
        for region in self.config.REGIONS:
            df_region = pd.DataFrame(index=self.years, columns=['Patients', 'Revenue (USD)', 'Market Share'])
            base_patients = self.regional_data.loc[region, 'Current Patients']
            growth_factor = 1 + (self.regional_data.loc[region, 'Growth Index'] * 0.1)
            df_region['Patients'] = [base_patients * (growth_factor ** year) for year in self.years]
            df_region['Revenue (USD)'] = df_region['Patients'] * self.config.PACKS_PER_PATIENT * self.sales_data['Price']
            df_region['Market Share'] = np.where(
                self.regional_data.loc[region, 'Patient Pool'] > 0,
                df_region['Patients'] / self.regional_data.loc[region, 'Patient Pool'],
                0
            )
            self.regional_time_series[region] = df_region

    def _calculate_competitive_metrics(self):
        """Calculates competitive market metrics"""
        self.competitive_data = pd.DataFrame(index=self.years)

        self.competitive_data['Our Share'] = self.sales_data['Segment Share']
        for competitor, shares in self.config.COMPETITOR_SHARES.items():
            comp_sh = shares.iloc[:len(self.years)].copy()
            self.competitive_data[f'{competitor} Share'] = comp_sh

        # Normalize shares if sum > 1
        share_cols = [c for c in self.competitive_data.columns if 'Share' in c]
        total_share = self.competitive_data[share_cols].sum(axis=1)
        for i in self.competitive_data.index:
            s = total_share[i]
            if s > 1:
                self.competitive_data.loc[i, share_cols] = self.competitive_data.loc[i, share_cols] / s

        self.competitive_data['Our Price (USD)'] = self.sales_data['Price']
        for competitor in self.config.COMPETITOR_SHARES.keys():
            price_factor = self.config.COMPETITOR_PRICE_FACTORS.get(competitor, 1.0)
            self.competitive_data[f'{competitor} Price (USD)'] = self.competitive_data['Our Price (USD)'] * price_factor

        def calculate_hhi(row):
            comp_shares = [row['Our Share']] + [row[f'{comp} Share'] for comp in self.config.COMPETITOR_SHARES.keys()]
            return sum((share ** 2) * 10000 for share in comp_shares)

        self.competitive_data['HHI'] = self.competitive_data.apply(calculate_hhi, axis=1)

        total_market = self.market_data * self.config.SEGMENT_SHARE
        self.competitive_data['Market Size (USD)'] = total_market

        competitors_with_prices = list(self.config.COMPETITOR_SHARES.keys())
        price_numerator = pd.Series(0.0, index=self.years)
        total_sh = pd.Series(0.0, index=self.years)

        for comp in competitors_with_prices:
            price_numerator += self.competitive_data[f'{comp} Price (USD)'] * self.competitive_data[f'{comp} Share']
            total_sh += self.competitive_data[f'{comp} Share']

        avg_market_price = price_numerator / total_sh
        self.competitive_data['Price Index'] = np.where(
            avg_market_price > 0,
            self.competitive_data['Our Price (USD)'] / avg_market_price,
            1
        )

    def calculate_npv(self):
        """Calculates NPV and related metrics in USD"""
        terminal_growth = 0.02
        fcf = self.cash_flows['FCF'].values
        wacc = self.config.WACC

        terminal_value = (fcf[-1] * (1 + terminal_growth)) / (wacc - terminal_growth)
        terminal_pv = terminal_value / ((1 + wacc) ** self.config.FORECAST_HORIZON)

        npv = npf.npv(wacc, fcf[1:]) + fcf[0] + terminal_pv
        irr = npf.irr(fcf)
        
        if irr is not None and np.isnan(irr):
            irr = None

        cumulative_fcf = np.cumsum(fcf)
        payback_period = None
        positive_mask = cumulative_fcf >= 0
        if positive_mask.any():
            payback_period = self.years[positive_mask][0]

        return {
            'NPV': npv / 1e6,  # Millions of USD
            'IRR': irr,
            'Terminal Value': terminal_value / 1e6,
            'Terminal Value PV': terminal_pv / 1e6,
            'Payback Period': payback_period
        }

    def _run_sensitivity_analysis(self):
        """Runs sensitivity analysis on key parameters"""
        original_values = {
            'PRICE_USD': self.config.PRICE_USD,
            'PRODUCT_SHARE_TARGET': self.config.PRODUCT_SHARE_TARGET,
            'VARIABLE_COST_USD': self.config.VARIABLE_COST_USD,
            'MARKET_GROWTH': self.config.MARKET_GROWTH,
            'SEGMENT_SHARE': self.config.SEGMENT_SHARE
        }
        
        sensitivity_params = {
            'PRICE_USD': {'base': self.config.PRICE_USD, 'range': (0.8, 1.2)},
            'PRODUCT_SHARE_TARGET': {'base': self.config.PRODUCT_SHARE_TARGET, 'range': (0.8, 1.2)},
            'VARIABLE_COST_USD': {'base': self.config.VARIABLE_COST_USD, 'range': (0.8, 1.2)},
            'MARKET_GROWTH': {'base': self.config.MARKET_GROWTH, 'range': (0.8, 1.2)},
            'SEGMENT_SHARE': {'base': self.config.SEGMENT_SHARE, 'range': (0.8, 1.2)}
        }
        
        base_npv = self.calculate_npv()['NPV']
        results = []
        for param, details in sensitivity_params.items():
            param_range = np.linspace(
                details['base'] * details['range'][0],
                details['base'] * details['range'][1],
                5
            )
            
            for value in param_range:
                setattr(self.config, param, value)
                self.run_model()
                npv = self.calculate_npv()['NPV']
                change = (npv - base_npv) / base_npv if base_npv != 0 else np.nan
                results.append({
                    'parameter': param,
                    'value': value,
                    'npv': npv,
                    'change': change
                })
            
            # Restore original value
            setattr(self.config, param, original_values[param])
            self.run_model()

        return pd.DataFrame(results)

    def _run_monte_carlo(self):
        """Runs Monte Carlo simulation on key uncertain parameters."""
        param_distributions = {
            'price_factor': stats.triang(c=0.5, loc=0.9, scale=0.2),
            'volume_factor': stats.triang(c=0.5, loc=0.9, scale=0.2),
            'market_growth': stats.norm(loc=self.config.MARKET_GROWTH, scale=0.005),
            'cogs_factor': stats.triang(c=0.5, loc=0.9, scale=0.2),
            'promo_cost_factor': stats.triang(c=0.5, loc=0.9, scale=0.2),
            'penetration_factor': stats.triang(c=0.5, loc=0.9, scale=0.2)
        }

        npv_results = np.zeros(self.config.NUM_SIMULATIONS)
        param_values = pd.DataFrame(index=range(self.config.NUM_SIMULATIONS), columns=param_distributions.keys())

        # Store original configuration values
        orig_price_usd = self.config.PRICE_USD
        orig_market_growth = self.config.MARKET_GROWTH
        orig_variable_cost_usd = self.config.VARIABLE_COST_USD
        orig_product_share_target = self.config.PRODUCT_SHARE_TARGET
        orig_promo_cost_per_patient_usd = self.config.PROMO_COST_PER_PATIENT_USD

        with tqdm(total=self.config.NUM_SIMULATIONS, desc="Running Monte Carlo") as pbar:
            for i in range(self.config.NUM_SIMULATIONS):
                current_params = {
                    param: dist.rvs()
                    for param, dist in param_distributions.items()
                }
                param_values.iloc[i] = current_params

                self.config.PRICE_USD = orig_price_usd * current_params['price_factor']
                # Ensure market growth > 0
                self.config.MARKET_GROWTH = max(current_params['market_growth'], 0.01)
                self.config.VARIABLE_COST_USD = orig_variable_cost_usd * current_params['cogs_factor']
                self.config.PRODUCT_SHARE_TARGET = orig_product_share_target * current_params['penetration_factor']
                self.config.PROMO_COST_PER_PATIENT_USD = orig_promo_cost_per_patient_usd * current_params['promo_cost_factor']

                self.run_model()

                mc_npv = self.calculate_npv()['NPV']
                npv_results[i] = mc_npv
                pbar.update(1)

        # Restore original configuration values
        self.config.PRICE_USD = orig_price_usd
        self.config.MARKET_GROWTH = orig_market_growth
        self.config.VARIABLE_COST_USD = orig_variable_cost_usd
        self.config.PRODUCT_SHARE_TARGET = orig_product_share_target
        self.config.PROMO_COST_PER_PATIENT_USD = orig_promo_cost_per_patient_usd
        self.run_model()

        return {
            'npv_results': npv_results,
            'param_values': param_values,
            'statistics': {
                'mean': np.mean(npv_results),
                'median': np.median(npv_results),
                'std': np.std(npv_results),
                'percentile_5': np.percentile(npv_results, 5),
                'percentile_95': np.percentile(npv_results, 95),
                'prob_negative': (npv_results < 0).mean()
            }
        }

    def run_full_analysis(self):
        """Full analysis run (including Monte Carlo and Sensitivity)"""
        self.run_model()
        monte_carlo_results = self._run_monte_carlo()
        sensitivity_results = self._run_sensitivity_analysis()

        npv_metrics = self.calculate_npv()
        irr = npv_metrics.get('IRR', None)
        if irr is not None and np.isnan(irr):
            irr = None

        results = {
            'market_data': self.market_data,
            'patient_data': self.patient_data,
            'sales_data': self.sales_data,
            'financials': self.financials,
            'cash_flows': self.cash_flows,
            'regional_data': self.regional_data,
            'competitive_data': self.competitive_data,
            'monte_carlo_results': monte_carlo_results,
            'sensitivity_results': sensitivity_results,
            'npv': npv_metrics['NPV'],
            'irr': irr,
            'terminal_value': npv_metrics.get('Terminal Value'),
            'payback_period': npv_metrics.get('Payback Period'),
            'summary_metrics': {
                'final_market_share': self.sales_data['Segment Share'].iloc[-1],
                'peak_revenue': self.sales_data['Sales'].max(),
                'total_capex': self.financials['CAPEX'].sum(),
                'final_year_ebitda_margin': self.financials['EBITDA Margin'].iloc[-1],
                'avg_roic': self.financials['ROIC'].mean(),
                'patient_penetration': self.sales_data['Patients on Product'].iloc[-1] / self.patient_data.iloc[-1] if self.patient_data.iloc[-1] != 0 else 0
            }
        }
        return results


class ModelVisualizer:
    """Enhanced visualizer for model results (in USD)"""
    def __init__(self, results, config):
        self.results = results
        self.config = config
        self.financials = results['financials']
        self.sales_data = results['sales_data']
        self.patient_data = results['patient_data']
        self.years = self.sales_data.index.to_numpy()
        self.setup_plotting_style()

    @staticmethod
    def setup_plotting_style():
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.grid'] = True

    def create_all_visualizations(self):
        self.plot_market_expansion()
        self.plot_patient_dynamics()
        self.plot_regional_analysis()
        self.plot_competitive_analysis()
        self.plot_financial_metrics()
        self.plot_risk_analysis()

    def plot_market_expansion(self):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        self.sales_data['Segment Share'].plot(ax=axes[0,0], marker='o', color='blue')
        axes[0,0].set_title('Market Penetration')
        axes[0,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

        self.sales_data['Patients on Product'].plot(ax=axes[0,1], marker='o', color='green')
        axes[0,1].set_title('Patient Growth')
        axes[0,1].set_ylabel('Number of Patients')

        self.sales_data['Revenue Growth'].plot(ax=axes[1,0], kind='bar', color='purple')
        axes[1,0].set_title('Revenue Growth')
        axes[1,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

        self.sales_data['Revenue per Patient'].plot(ax=axes[1,1], marker='o', color='red')
        axes[1,1].set_title('Revenue per Patient (USD)')
        axes[1,1].set_ylabel('USD per Patient')
        
        plt.tight_layout()
        plt.show()

    def plot_patient_dynamics(self):
        promo_cost_per_patient_usd = self.config.PROMO_COST_PER_PATIENT_USD * ((1 + self.config.INFLATION_USD) ** self.years)
        promo_spend_usd = self.sales_data['Patients on Product'] * promo_cost_per_patient_usd

        new_patients = self.sales_data['Patients on Product'].diff().fillna(
            self.sales_data['Patients on Product'].iloc[0]
        )
        cumulative_new_patients = new_patients.cumsum()
        acquisition_cost_usd = np.where(self.sales_data['Patients on Product']>0, promo_spend_usd / self.sales_data['Patients on Product'], 0)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        pd.DataFrame({
            'Total Patients': self.patient_data,
            'Treated Patients': self.sales_data['Patients on Product']
        }).plot(ax=axes[0,0])
        axes[0,0].set_title('Patient Pool vs Treated Patients')
        axes[0,0].set_ylabel('Number of Patients')

        treatment_rate = np.where(
            self.patient_data > 0,
            self.sales_data['Patients on Product'] / self.patient_data,
            0
        )
        pd.Series(treatment_rate, index=self.patient_data.index).plot(ax=axes[0,1], marker='o', color='green')
        axes[0,1].set_title('Treatment Rate')
        axes[0,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

        pd.Series(acquisition_cost_usd, index=self.sales_data.index).plot(ax=axes[1,0], marker='o', color='red')
        axes[1,0].set_title('Patient Acquisition Cost (USD per Patient)')
        axes[1,0].set_ylabel('USD per Patient')

        cumulative_new_patients.plot(ax=axes[1,1], color='purple')
        axes[1,1].set_title('Cumulative New Patients')
        axes[1,1].set_ylabel('Number of Patients')

        plt.tight_layout()
        plt.show()

    def plot_regional_analysis(self):
        regional_data = self.results['regional_data'].copy()
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        regional_data['Patient Pool'].plot(kind='bar', ax=axes[0,0], color='blue')
        axes[0,0].set_title('Regional Market Size (Patients)')
        axes[0,0].set_ylabel('Number of Patients')

        regional_data['Penetration Rate'].plot(kind='bar', ax=axes[0,1], color='green')
        axes[0,1].set_title('Regional Penetration Rate')
        axes[0,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

        regional_data['Revenue Potential (USD)'].plot(kind='bar', ax=axes[1,0], color='purple')
        axes[1,0].set_title('Regional Revenue Potential (USD)')
        axes[1,0].set_ylabel('USD')

        regional_data['Growth Index'].plot(kind='bar', ax=axes[1,1], color='red')
        axes[1,1].set_title('Growth Opportunity Index')
        
        plt.tight_layout()
        plt.show()

    def plot_competitive_analysis(self):
        competitive_data = self.results['competitive_data'].copy()
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        share_columns = [col for col in competitive_data.columns if 'Share' in col and 'Price' not in col]
        competitive_data[share_columns].plot(ax=axes[0,0], title='Market Share Evolution')
        axes[0,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

        price_cols = [c for c in competitive_data.columns if '(USD)' in c and 'Price' in c]
        if price_cols:
            # Just plot all price columns
            competitive_data[price_cols].plot(ax=axes[0,1], title='Price Positioning (USD)')
            axes[0,1].set_ylabel('Price (USD)')
        
        if 'HHI' in competitive_data.columns:
            competitive_data['HHI'].plot(ax=axes[1,0], color='purple', title='Market Concentration (HHI)')
            axes[1,0].set_ylabel('HHI Index')
        
        if 'Price Index' in competitive_data.columns:
            competitive_data['Price Index'].plot(ax=axes[1,1], color='red', title='Relative Price Index')
            axes[1,1].axhline(y=1, color='black', linestyle='--')
            axes[1,1].set_ylabel('Price Index')
        
        plt.tight_layout()
        plt.show()

    def plot_financial_metrics(self):
        financials = self.financials.copy()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        profit_metrics = pd.DataFrame({
            'Gross Profit': financials['Gross Profit']/1e6,
            'EBITDA': financials['EBITDA']/1e6,
            'Net Income': financials['Net Income']/1e6
        }, index=financials.index)
        
        profit_metrics.plot(ax=axes[0,0], title='Profitability Metrics (Millions of USD)')
        axes[0,0].set_ylabel('Million USD')
        
        margin_metrics = pd.DataFrame({
            'Gross Margin': financials['Gross Margin'],
            'EBITDA Margin': financials['EBITDA Margin'],
            'Net Margin': np.where(financials['Sales']>0, financials['Net Income']/financials['Sales'], 0)
        }, index=financials.index)
        
        margin_metrics.plot(ax=axes[0,1], title='Margin Evolution')
        axes[0,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        capex_data = pd.DataFrame({
            'Expansion CAPEX': financials['Expansion CAPEX']/1e6,
            'Maintenance CAPEX': financials['Maintenance CAPEX']/1e6
        }, index=financials.index)
        
        capex_data.plot(kind='bar', stacked=True, ax=axes[1,0], title='CAPEX Breakdown (Millions of USD)')
        axes[1,0].set_ylabel('Million USD')
        
        roi_metrics = pd.DataFrame({'ROIC': financials['ROIC']}, index=financials.index)
        roi_metrics.plot(ax=axes[1,1], title='Return Metrics')
        axes[1,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        plt.tight_layout()
        plt.show()

        cash_flows = self.results['cash_flows'].copy()
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        cash_flow_data = pd.DataFrame({
            'Operating CF': cash_flows['CFO']/1e6,
            'Investment CF': cash_flows['CFI']/1e6,
            'Free CF': cash_flows['FCF']/1e6
        }, index=cash_flows.index)
        
        cash_flow_data.plot(ax=axes[0,0], title='Cash Flow Evolution (Millions of USD)')
        axes[0,0].set_ylabel('Million USD')
        
        (cash_flows['FCF'].cumsum() / 1e6).plot(ax=axes[0,1], title='Cumulative Free Cash Flow (Millions of USD)', color='green')
        axes[0,1].set_ylabel('Million USD')
        
        working_capital = pd.DataFrame({
            'Change in Working Capital': self.financials['Change in NWC']/1e6,
            'Cumulative Working Capital': self.financials['Change in NWC'].cumsum()/1e6
        }, index=financials.index)
        
        working_capital.plot(ax=axes[1,0], title='Working Capital Evolution (Millions of USD)')
        axes[1,0].set_ylabel('Million USD')
        
        cost_structure = pd.DataFrame({
            'COGS': financials['COGS']/1e6,
            'SG&A': financials['SG&A']/1e6,
            'D&A': financials['D&A']/1e6
        }, index=financials.index)
        
        cost_structure.plot(kind='area', stacked=True, ax=axes[1,1], title='Cost Structure (Millions of USD)')
        axes[1,1].set_ylabel('Million USD')
        
        plt.tight_layout()
        plt.show()

    def plot_risk_analysis(self):
        monte_carlo = self.results['monte_carlo_results']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        npv_results = monte_carlo['npv_results']
        sns.histplot(npv_results, kde=True, ax=axes[0,0], color='blue')
        mean_val = np.mean(npv_results)
        axes[0,0].axvline(mean_val, color='red', linestyle='--', label=f"Mean: {mean_val:.1f}M USD")
        axes[0,0].set_title('NPV Distribution (Millions of USD)')
        axes[0,0].set_xlabel('NPV (Millions of USD)')
        axes[0,0].legend()

        correlations = monte_carlo['param_values'].corrwith(pd.Series(monte_carlo['npv_results']))
        correlations.plot(kind='barh', ax=axes[0,1], color='green')
        axes[0,1].set_title('Sensitivity Analysis')
        axes[0,1].set_xlabel('Correlation with NPV (USD)')

        sorted_npv = np.sort(npv_results)
        probabilities = np.arange(1, len(sorted_npv) + 1) / len(sorted_npv)
        axes[1,0].plot(sorted_npv, probabilities, color='purple')
        axes[1,0].set_title('Value at Risk (USD)')
        axes[1,0].set_xlabel('NPV (Millions of USD)')
        axes[1,0].set_ylabel('Cumulative Probability')

        scenarios = pd.Series({
            'Worst Case': np.percentile(npv_results, 5),
            'Base Case': np.median(npv_results),
            'Best Case': np.percentile(npv_results, 95)
        })
        scenarios.plot(kind='bar', ax=axes[1,1], color=['red', 'blue', 'green'])
        axes[1,1].set_title('Scenario Analysis (USD)')
        axes[1,1].set_ylabel('NPV (Millions of USD)')

        plt.tight_layout()
        plt.show()


class ResultsExporter:
    """Class for exporting model results."""

    def __init__(self, results, config):
        """
        Initializes the exporter.

        Args:
            results (dict): Model results.
            config (ModelConfig): Model configuration.
        """
        self.results = results
        self.config = config

    def export_to_excel(self, filename='model_results_usd.xlsx'):
        """
        Exports results to an Excel file.

        Args:
            filename (str, optional): The filename. Defaults to 'model_results_usd.xlsx'.
        """
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            self.results['sales_data'].to_excel(writer, sheet_name='Sales Data')
            self.results['financials'].to_excel(writer, sheet_name='Financials')
            self.results['cash_flows'].to_excel(writer, sheet_name='Cash Flows')
            self.results['regional_data'].to_excel(writer, sheet_name='Regional Analysis')
            self.results['competitive_data'].to_excel(writer, sheet_name='Competitive Analysis')
            pd.DataFrame({'NPV Results (Millions of USD)': self.results['monte_carlo_results']['npv_results']}).to_excel(writer, sheet_name='Monte Carlo')
            self.results['sensitivity_results'].to_excel(writer, sheet_name='Sensitivity')
        print(f"Results successfully exported to {filename}")

    def generate_summary_report(self):
        """
        Generates a summary report of key metrics.

        Returns:
            pd.DataFrame: Summary report.
        """
        summary = {
            'Market Metrics': {
                'Final Market Share': self.results['sales_data']['Segment Share'].iloc[-1],
                'Total Patients': self.results['sales_data']['Patients on Product'].iloc[-1],
                'Revenue (USD)': self.results['sales_data']['Sales'].iloc[-1]
            },
            'Financial Metrics': {
                'NPV (Millions of USD)': self.results['npv'],
                'IRR': self.results['irr'],
                'Payback Period': self.results['payback_period'],
                'Final Year EBITDA Margin': self.results['financials']['EBITDA Margin'].iloc[-1],
                'Average ROIC': self.results['summary_metrics']['avg_roic']
            },
            'Risk Metrics': {
                'NPV Mean (Millions of USD)': self.results['monte_carlo_results']['statistics']['mean'],
                'NPV Std Dev (Millions of USD)': self.results['monte_carlo_results']['statistics']['std'],
                'Probability of Negative NPV': self.results['monte_carlo_results']['statistics']['prob_negative']
            }
        }
        summary_df = pd.DataFrame(summary)
        return summary_df



def main():
    try:
        config = ModelConfig()
        model = FinancialModel(config)
        
        print("Running full analysis...")
        results = model.run_full_analysis()

        print("Generating visualizations...")
        visualizer = ModelVisualizer(results, config)
        visualizer.create_all_visualizations()

        print("Exporting results in USD...")
        exporter = ResultsExporter(results, config)
        exporter.export_to_excel()
        summary = exporter.generate_summary_report()
        
        print("\nAnalysis Complete!")
        print("\nKey Metrics Summary (in USD):")
        print(summary)

    except Exception as e:
        print("An error occurred during the financial model execution.")
        print(str(e))
        raise e


if __name__ == "__main__":
    main()
