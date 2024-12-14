# Install necessary packages
!pip install numpy numpy_financial pandas seaborn matplotlib scipy tqdm scikit-learn openpyxl

# Import necessary libraries
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
    """Configuration class for financial model parameters"""
    def __init__(self):
        # Market Parameters
        self.POPULATION = 87.5e6
        self.GDP_GROWTH = 0.12  # Annual GDP growth rate
        self.INFLATION = 0.45   # Annual inflation rate (Adjusted from 45% to 4%)
        
        # Exchange Rate (0-th year)
        self.FX_RATE = 35.0
        
        self.MARKET_VOLUME_USD = 8.0e9
        self.MARKET_GROWTH = 0.045
        self.SEGMENT_SHARE = 0.176

        # Product Parameters (Converted to TRY)
        self.DOSAGE_MG = 2000
        self.PACKS_PER_PATIENT = 24
        self.PRICE_USD = 300
        self.PRODUCT_SHARE_TARGET = 0.8
        self.PRODUCT_RAMP_UP = np.array([0, 0.2, 0.6, 0.8, 1, 1, 1, 1])

        # Patient Parameters
        self.PREVALENCE = 2160
        self.NEW_CASES_RATE = 0.0000416
        self.DIAGNOSIS_RATE = 0.3
        self.ELIGIBILITY_RATE = 0.75
        self.MORTALITY_RATE = 0.017

        # Financial Parameters (Converted to TRY)
        self.VARIABLE_COST_USD = 20
        self.SGA_RATE = 0.35
        self.CAPEX_STARTUP_USD = 2e6
        self.CAPEX_ADDITIONAL_PER_10K_PACKS_USD = 1e6
        self.PROMO_COST_PER_PATIENT_USD = 200
        self.WORKING_CAPITAL_RATE = 0.12
        self.TAX_RATE = 0.25
        self.WACC = 0.10
        self.FORECAST_HORIZON = 7
        self.NUM_SIMULATIONS = 10000

        # Regional Parameters
        self.REGIONS = ['North', 'South', 'East', 'West', 'Central']
        self.REGION_POPULATIONS = np.array([20, 25, 15, 18, 22]) * 1e6
        self.REGION_PREVALENCE = np.array([0.0020, 0.0025, 0.0018, 0.0022, 0.0021])
        self.REGION_PENETRATION = np.array([0.15, 0.12, 0.08, 0.10, 0.11])

        # Competitive Parameters
        self.COMPETITOR_SHARES = {
            'Competitor 1': pd.Series([0.3, 0.28, 0.25, 0.23, 0.2, 0.18, 0.15, 0.12]),
            'Competitor 2': pd.Series([0.2, 0.19, 0.18, 0.17, 0.15, 0.14, 0.12, 0.1]),
            'Others':       pd.Series([0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15])
        }
        self.COMPETITOR_PRICE_FACTORS = {
            'Competitor 1': 1.1,  # 10% premium
            'Competitor 2': 0.9   # 10% discount
        }

    def convert_to_lira(self):
        """Converts all USD-based parameters to TRY using the FX rate"""
        self.MARKET_VOLUME = self.MARKET_VOLUME_USD * self.FX_RATE

        self.PRICE_LIRA_0 = self.PRICE_USD * self.FX_RATE
        self.VARIABLE_COST_LIRA_0 = self.VARIABLE_COST_USD * self.FX_RATE
        self.CAPEX_STARTUP_LIRA_0 = self.CAPEX_STARTUP_USD * self.FX_RATE
        self.CAPEX_ADDITIONAL_PER_10K_PACKS_LIRA_0 = self.CAPEX_ADDITIONAL_PER_10K_PACKS_USD * self.FX_RATE
        self.PROMO_COST_PER_PATIENT_LIRA_0 = self.PROMO_COST_PER_PATIENT_USD * self.FX_RATE

        # Dynamic Variables for Sensitivity and Monte Carlo
        self.PRICE_LIRA = self.PRICE_LIRA_0
        self.VARIABLE_COST = self.VARIABLE_COST_LIRA_0
        self.PROMO_COST_PER_PATIENT = self.PROMO_COST_PER_PATIENT_LIRA_0
        self.CAPEX_STARTUP = self.CAPEX_STARTUP_LIRA_0
        self.CAPEX_ADDITIONAL_PER_10K_PACKS = self.CAPEX_ADDITIONAL_PER_10K_PACKS_LIRA_0

class FinancialModel:
    """Main financial model class"""
    def __init__(self, config=None):
        self.config = config or ModelConfig()
        self.config.convert_to_lira()
        
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
        """Calculates the forecasted market volume over the forecast horizon"""
        # Adjusted: Align array length with forecast horizon (excluding year 0)
        gdp_growth = self.config.GDP_GROWTH * np.ones(self.config.FORECAST_HORIZON)  # Changed from +1 to match forecast steps
        macro_adjust = 1 + 0.8 * gdp_growth - 0.6 * self.config.INFLATION
        market_growth = self.config.MARKET_GROWTH * macro_adjust

        market_outlook_values = [self.config.MARKET_VOLUME]
        for growth in market_growth:
            new_volume = market_outlook_values[-1] * (1 + growth)
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
            patients[i] = max(patients[i-1] + eligible - deaths, 0)  # Prevent negative patients

        self.patient_data = pd.Series(patients, index=self.years)

    def _calculate_sales_forecast(self):
        """Calculates sales forecast based on patient data and market conditions"""
        self.sales_data = pd.DataFrame(index=self.years)
        
        # Patients on Product considering ramp-up
        self.sales_data['Patients on Product'] = (
            self.patient_data * 
            self.config.PRODUCT_SHARE_TARGET * 
            self.config.PRODUCT_RAMP_UP
        )
        
        self.sales_data['Sales Volume, packs'] = (
            self.sales_data['Patients on Product'] * 
            self.config.PACKS_PER_PATIENT
        )
        
        # Price in TRY with inflation
        self.sales_data['Price'] = (
            self.config.PRICE_LIRA * 
            (1 + self.config.INFLATION) ** self.years
        )
        
        self.sales_data['Sales'] = (
            self.sales_data['Sales Volume, packs'] * self.sales_data['Price']
        )
        
        segment_value = self.market_data * self.config.SEGMENT_SHARE
        self.sales_data['Segment Share'] = self.sales_data['Sales'].divide(segment_value, fill_value=0).clip(upper=1.0)
        
        self.sales_data['Revenue Growth'] = self.sales_data['Sales'].pct_change().fillna(0)
        self.sales_data['Volume Growth'] = self.sales_data['Sales Volume, packs'].pct_change().fillna(0)
        self.sales_data['Patient Growth'] = self.sales_data['Patients on Product'].pct_change().fillna(0)
        
        self.sales_data['Revenue per Patient'] = np.where(
            self.sales_data['Patients on Product'] > 0,
            self.sales_data['Sales'] / self.sales_data['Patients on Product'],
            0
        )

        # Ensure no NaNs in year 0
        self.sales_data.loc[0, [
            'Sales', 
            'Sales Volume, packs',
            'Patients on Product',
            'Segment Share',
            'Revenue Growth',
            'Volume Growth',
            'Patient Growth'
        ]] = self.sales_data.loc[0, [
            'Sales', 
            'Sales Volume, packs',
            'Patients on Product',
            'Segment Share',
            'Revenue Growth',
            'Volume Growth',
            'Patient Growth'
        ]].fillna(0)

    def _calculate_financials(self):
        """Calculates financial metrics over the forecast horizon"""
        self.financials = pd.DataFrame(index=self.years)
        
        self.financials['Sales'] = self.sales_data['Sales']

        # COGS grows only with inflation
        cogs_per_pack = (
            self.config.VARIABLE_COST * 
            (1 + self.config.INFLATION) ** self.financials.index
        )

        self.financials['COGS'] = (
            self.sales_data['Sales Volume, packs'] * cogs_per_pack
        )
        
        self.financials['Gross Profit'] = self.financials['Sales'] - self.financials['COGS']
        self.financials['Gross Margin'] = np.where(
            self.financials['Sales'] > 0,
            self.financials['Gross Profit'] / self.financials['Sales'],
            0
        )
        
        # Integrate promo expenses into SG&A
        base_sg_a = self.financials['Sales'] * self.config.SGA_RATE
        promo_cost_yearly = (1 + self.config.INFLATION) ** self.years * self.config.PROMO_COST_PER_PATIENT
        promo_spend = self.sales_data['Patients on Product'] * promo_cost_yearly
        self.financials['SG&A'] = base_sg_a + promo_spend
        
        self.financials['EBITDA'] = self.financials['Gross Profit'] - self.financials['SG&A']
        self.financials['EBITDA Margin'] = np.where(
            self.financials['Sales'] > 0,
            self.financials['EBITDA'] / self.financials['Sales'],
            0
        )

        # CAPEX
        expansion_capex = np.zeros_like(self.years, dtype=float)
        expansion_capex[0] = self.config.CAPEX_STARTUP

        additional_capex = (
            np.ceil(self.sales_data['Sales Volume, packs'] / 10000) * 
            (self.config.CAPEX_ADDITIONAL_PER_10K_PACKS * 
             (1 + self.config.INFLATION) ** self.years)
        )

        maintenance_capex = self.financials['Sales'] * 0.03

        self.financials['Expansion CAPEX'] = expansion_capex
        self.financials['Additional CAPEX'] = additional_capex
        self.financials['Maintenance CAPEX'] = maintenance_capex
        
        self.financials['CAPEX'] = (
            self.financials['Expansion CAPEX'] + 
            self.financials['Additional CAPEX'] + 
            self.financials['Maintenance CAPEX']
        )
        
        # Depreciation & Amortization (D&A) over 7 years (Adjusted from 10)
        horizon = self.config.FORECAST_HORIZON + 1
        da = np.zeros(horizon)
        for i in range(horizon):
            capex_i = self.financials['CAPEX'].iloc[i]
            end_year = min(i + 7, horizon)  # Depreciate over 7 years
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
        
        net_working_capital = (
            self.financials['Sales'] * self.config.WORKING_CAPITAL_RATE
        )
        self.financials['Change in NWC'] = net_working_capital.diff().fillna(0)
        
        invested_capital = self.financials['CAPEX'].cumsum() + net_working_capital
        self.financials['ROIC'] = np.where(
            invested_capital != 0,
            (self.financials['EBIT'] * (1 - self.config.TAX_RATE)) / invested_capital,
            0
        )
        
        # Removed ROE calculation due to lack of equity modeling

    def _calculate_cash_flows(self):
        """Calculates cash flows based on financial metrics"""
        self.cash_flows = pd.DataFrame(index=self.years)
        
        self.cash_flows['EBITDA'] = self.financials['EBITDA']
        self.cash_flows['D&A'] = self.financials['D&A']
        self.cash_flows['Tax'] = self.financials['Tax']
        self.cash_flows['Change in NWC'] = self.financials['Change in NWC']
        
        # CFO includes D&A
        self.cash_flows['CFO'] = (
            self.financials['EBITDA'] - 
            self.cash_flows['Tax'] - 
            self.cash_flows['Change in NWC'] +
            self.financials['D&A']
        )
        
        self.cash_flows['Expansion CAPEX'] = self.financials['Expansion CAPEX']
        self.cash_flows['Additional CAPEX'] = self.financials['Additional CAPEX']
        self.cash_flows['Maintenance CAPEX'] = self.financials['Maintenance CAPEX']
        
        self.cash_flows['CFI'] = (
            -self.financials['Expansion CAPEX'] - 
            self.financials['Additional CAPEX'] - 
            self.financials['Maintenance CAPEX']
        )
        
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
        """Calculates regional market metrics"""
        self.regional_data = pd.DataFrame(
            index=self.config.REGIONS,
            columns=[
                'Population',
                'Patient Pool',
                'Current Patients',
                'Penetration Rate',
                'Revenue Potential (Lira)',
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
        
        # Revenue Potential using base PRICE_LIRA without inflation
        self.regional_data['Revenue Potential (Lira)'] = (
            self.regional_data['Patient Pool'] *
            self.config.PRODUCT_SHARE_TARGET *
            self.config.PACKS_PER_PATIENT *
            self.config.PRICE_LIRA
        )
        
        self.regional_data['Growth Index'] = (
            (1 - self.regional_data['Penetration Rate']) *
            (self.config.REGION_PREVALENCE / self.config.REGION_PREVALENCE.mean())
        )
        
        self.regional_time_series = {}
        for region in self.config.REGIONS:
            self.regional_time_series[region] = pd.DataFrame(
                index=self.years,
                columns=['Patients', 'Revenue (Lira)', 'Market Share']
            )
            
            base_patients = self.regional_data.loc[region, 'Current Patients']
            growth_factor = 1 + (self.regional_data.loc[region, 'Growth Index'] * 0.1)
            
            self.regional_time_series[region]['Patients'] = [
                base_patients * (growth_factor ** year) for year in self.years
            ]
            
            self.regional_time_series[region]['Revenue (Lira)'] = (
                self.regional_time_series[region]['Patients'] *
                self.config.PACKS_PER_PATIENT *
                self.sales_data['Price']
            )
            
            self.regional_time_series[region]['Market Share'] = np.where(
                self.regional_data.loc[region, 'Patient Pool'] > 0,
                self.regional_time_series[region]['Patients'] / self.regional_data.loc[region, 'Patient Pool'],
                0
            )

    def _calculate_competitive_metrics(self):
        """Calculates competitive market metrics"""
        self.competitive_data = pd.DataFrame(index=self.years)
        
        self.competitive_data['Our Share'] = self.sales_data['Segment Share']
        for competitor, shares in self.config.COMPETITOR_SHARES.items():
            self.competitive_data[f'{competitor} Share'] = shares
        
        self.competitive_data['Our Price (Lira)'] = self.sales_data['Price']
        
        for competitor in self.config.COMPETITOR_SHARES.keys():
            price_factor = self.config.COMPETITOR_PRICE_FACTORS.get(competitor, 1.0)
            self.competitive_data[f'{competitor} Price (Lira)'] = self.competitive_data['Our Price (Lira)'] * price_factor
        
        def calculate_hhi(row):
            comp_shares = [row['Our Share']] + [row[f'{comp} Share'] for comp in self.config.COMPETITOR_SHARES.keys()]
            return sum((share ** 2) * 10000 for share in comp_shares)
        
        self.competitive_data['HHI'] = self.competitive_data.apply(calculate_hhi, axis=1)
        
        total_market = self.market_data * self.config.SEGMENT_SHARE
        self.competitive_data['Market Size (Lira)'] = total_market
        
        competitors_with_prices = list(self.config.COMPETITOR_SHARES.keys())
        price_numerator = pd.Series(0.0, index=self.years)
        total_share = pd.Series(0.0, index=self.years)
        
        for comp in competitors_with_prices:
            price_numerator += self.competitive_data[f'{comp} Price (Lira)'] * self.competitive_data[f'{comp} Share']
            total_share += self.competitive_data[f'{comp} Share']
        
        avg_market_price = self.competitive_data['Our Price (Lira)'].copy()
        mask = total_share > 0
        avg_market_price[mask] = price_numerator[mask] / total_share[mask]
        
        self.competitive_data['Price Index'] = np.where(
            avg_market_price > 0,
            self.competitive_data['Our Price (Lira)'] / avg_market_price,
            1
        )

    def calculate_npv(self):
        """Calculates Net Present Value (NPV) and other related metrics"""
        terminal_growth = 0.02
        fcf = self.cash_flows['FCF'].values
        wacc = self.config.WACC

        # Calculate Terminal Value using Gordon Growth Model
        terminal_value = (fcf[-1] * (1 + terminal_growth)) / (wacc - terminal_growth)
        terminal_pv = terminal_value / ((1 + wacc) ** self.config.FORECAST_HORIZON)

        # NPV Calculation
        # Ensure fcf[0] is included correctly (initial investment should be negative if applicable)
        npv = npf.npv(wacc, fcf[1:]) + fcf[0] + terminal_pv

        # IRR Calculation without Terminal Value
        # Exclude terminal value to prevent skewed IRR
        irr = npf.irr(fcf)
        
        # Handle cases where IRR cannot be computed
        if irr is not None and np.isnan(irr):
            irr = None

        # Payback Period Calculation
        cumulative_fcf = np.cumsum(fcf)
        payback_period = None
        positive_mask = cumulative_fcf >= 0
        if positive_mask.any():
            payback_period = self.years[positive_mask][0]

        return {
            'NPV': npv * 1e-6,  # Millions of Lira
            'IRR': irr,
            'Terminal Value': terminal_value * 1e-6,
            'Terminal Value PV': terminal_pv * 1e-6,
            'Payback Period': payback_period
        }

    def get_results(self, monte_carlo_results, sensitivity_results):
        """Compiles all results into a dictionary"""
        npv_metrics = self.calculate_npv()
        
        irr = npv_metrics.get('IRR', None)
        if irr is not None and np.isnan(irr):
            irr = None

        return {
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

    def run_full_analysis(self):
        """Executes the full financial model analysis"""
        print("Running full model analysis...")
        self.run_model()
        
        print("Running Monte Carlo simulation...")
        monte_carlo_results = self._run_monte_carlo()
        
        print("Calculating sensitivity analysis...")
        sensitivity_results = self._run_sensitivity_analysis()
        
        return self.get_results(monte_carlo_results, sensitivity_results)

    def _run_sensitivity_analysis(self):
        """Performs sensitivity analysis on key parameters"""
        sensitivity_params = {
            'PRICE_LIRA': {'base': self.config.PRICE_LIRA, 'range': (0.8, 1.2)},
            'PRODUCT_SHARE_TARGET': {'base': self.config.PRODUCT_SHARE_TARGET, 'range': (0.8, 1.2)},
            'VARIABLE_COST': {'base': self.config.VARIABLE_COST, 'range': (0.8, 1.2)},
            'MARKET_GROWTH': {'base': self.config.MARKET_GROWTH, 'range': (0.8, 1.2)},
            'SEGMENT_SHARE': {'base': self.config.SEGMENT_SHARE, 'range': (0.8, 1.2)}
        }
        
        original_values = {}
        for param, details in sensitivity_params.items():
            original_values[param] = getattr(self.config, param)
        
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
        """Performs Monte Carlo simulation to assess risk"""
        param_distributions = {
            'price_factor': stats.triang(c=0.5, loc=0.9, scale=0.2),            # 0.9 to 1.1
            'volume_factor': stats.triang(c=0.5, loc=0.9, scale=0.2),           # 0.9 to 1.1
            'market_growth': stats.norm(loc=self.config.MARKET_GROWTH, scale=0.005),
            'cogs_factor': stats.triang(c=0.5, loc=0.9, scale=0.2),             # 0.9 to 1.1
            'promo_cost_factor': stats.triang(c=0.5, loc=0.9, scale=0.2),       # 0.9 to 1.1
            'penetration_factor': stats.triang(c=0.5, loc=0.9, scale=0.2)      # 0.9 to 1.1
        }

        npv_results = np.zeros(self.config.NUM_SIMULATIONS)
        param_values = pd.DataFrame(index=range(self.config.NUM_SIMULATIONS), columns=param_distributions.keys())

        # Store original configuration values
        orig_price_lira = self.config.PRICE_LIRA
        orig_market_growth = self.config.MARKET_GROWTH
        orig_variable_cost = self.config.VARIABLE_COST
        orig_product_share_target = self.config.PRODUCT_SHARE_TARGET
        orig_promo_cost_per_patient = self.config.PROMO_COST_PER_PATIENT

        with tqdm(total=self.config.NUM_SIMULATIONS, desc="Running Monte Carlo") as pbar:
            for i in range(self.config.NUM_SIMULATIONS):
                current_params = {
                    param: dist.rvs()
                    for param, dist in param_distributions.items()
                }
                param_values.iloc[i] = current_params
                
                # Apply parameter adjustments
                self.config.PRICE_LIRA = orig_price_lira * current_params['price_factor']
                self.config.MARKET_GROWTH = max(current_params['market_growth'], 0.01)  # Prevent negative growth
                self.config.VARIABLE_COST = orig_variable_cost * current_params['cogs_factor']
                self.config.PRODUCT_SHARE_TARGET = orig_product_share_target * current_params['penetration_factor']
                self.config.PROMO_COST_PER_PATIENT = orig_promo_cost_per_patient * current_params['promo_cost_factor']

                self.run_model()

                mc_npv = self.calculate_npv()['NPV']
                npv_results[i] = mc_npv
                pbar.update(1)

        # Restore original configuration values
        self.config.PRICE_LIRA = orig_price_lira
        self.config.MARKET_GROWTH = orig_market_growth
        self.config.VARIABLE_COST = orig_variable_cost
        self.config.PRODUCT_SHARE_TARGET = orig_product_share_target
        self.config.PROMO_COST_PER_PATIENT = orig_promo_cost_per_patient
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

class ModelVisualizer:
    """Enhanced visualizer for model results (all currency in Lira)"""
    def __init__(self, results, config):
        self.results = results
        self.config = config
        self.financials = results['financials']
        self.sales_data = results['sales_data']
        self.patient_data = results['patient_data']
        self.years = self.sales_data.index.to_numpy()  # Added to fix the AttributeError
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

    def plot_financial_metrics(self):
        """Plots financial metrics such as profits, margins, CAPEX, ROI, etc."""
        financials = self.results['financials']
        
        # Profit Metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        profit_metrics = pd.DataFrame({
            'Gross Profit': financials['Gross Profit'],
            'EBITDA': financials['EBITDA'],
            'Net Income': financials['Net Income']
        }) / 1e6
        
        profit_metrics.plot(ax=axes[0,0], title='Profitability Metrics (Millions of Lira)')
        axes[0,0].set_ylabel('Million Lira')
        
        # Margin Metrics
        margin_metrics = pd.DataFrame({
            'Gross Margin': financials['Gross Margin'],
            'EBITDA Margin': financials['EBITDA Margin'],
            'Net Margin': np.where(financials['Sales']>0, financials['Net Income']/financials['Sales'], 0)
        })
        
        margin_metrics.plot(ax=axes[0,1], title='Margin Evolution')
        axes[0,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        # CAPEX Breakdown
        capex_data = pd.DataFrame({
            'Expansion CAPEX': financials['Expansion CAPEX'],
            'Additional CAPEX': financials['Additional CAPEX'],
            'Maintenance CAPEX': financials['Maintenance CAPEX']
        }) / 1e6
        
        capex_data.plot(kind='bar', stacked=True, ax=axes[1,0], title='CAPEX Breakdown (Millions of Lira)')
        axes[1,0].set_ylabel('Million Lira')
        
        # Return Metrics
        roi_metrics = pd.DataFrame({
            'ROIC': financials['ROIC']
            # ROE removed
        })
        
        roi_metrics.plot(ax=axes[1,1], title='Return Metrics')
        axes[1,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        plt.tight_layout()
        plt.show()

        # Additional Financial Metrics Plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        cash_flows = self.results['cash_flows']
        cash_flow_data = pd.DataFrame({
            'Operating CF': cash_flows['CFO'],
            'Investment CF': cash_flows['CFI'],
            'Free CF': cash_flows['FCF']
        }) / 1e6
        
        cash_flow_data.plot(ax=axes[0,0], title='Cash Flow Evolution (Millions of Lira)')
        axes[0,0].set_ylabel('Million Lira')
        
        (cash_flows['FCF'].cumsum() / 1e6).plot(ax=axes[0,1], title='Cumulative Free Cash Flow (Millions of Lira)', color='green')
        axes[0,1].set_ylabel('Million Lira')
        
        working_capital = pd.DataFrame({
            'Change in Working Capital': self.financials['Change in NWC'],
            'Cumulative Working Capital': self.financials['Change in NWC'].cumsum()
        }) / 1e6
        
        working_capital.plot(ax=axes[1,0], title='Working Capital Evolution (Millions of Lira)')
        axes[1,0].set_ylabel('Million Lira')
        
        cost_structure = pd.DataFrame({
            'COGS': financials['COGS'],
            'SG&A': financials['SG&A'],
            'D&A': financials['D&A']
        }) / 1e6
        
        cost_structure.plot(kind='area', stacked=True, ax=axes[1,1], title='Cost Structure (Millions of Lira)')
        axes[1,1].set_ylabel('Million Lira')
        
        plt.tight_layout()
        plt.show()

        # Efficiency Metrics
        plt.figure(figsize=(12, 6))
        efficiency_metrics = pd.DataFrame({
            'Asset Turnover': np.where(
                self.financials['CAPEX'].cumsum() > 0, 
                self.financials['Sales'] / self.financials['CAPEX'].cumsum(), 
                0
            ),
            'Operating Efficiency': np.where(
                self.financials['Sales'] > 0, 
                self.financials['EBITDA'] / self.financials['Sales'], 
                0
            ),
            'Cash Conversion': np.where(
                self.financials['EBITDA'] > 0, 
                self.results['cash_flows']['CFO'] / self.financials['EBITDA'], 
                0
            )
        })
        
        efficiency_metrics.plot(title='Efficiency Metrics')
        plt.ylabel('Ratio')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_competitive_analysis(self):
        """Plots competitive market metrics such as market share evolution and price positioning"""
        competitive_data = self.results['competitive_data']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Market Share Evolution
        share_columns = [col for col in competitive_data.columns if 'Share' in col]
        competitive_data[share_columns].plot(ax=axes[0,0], title='Market Share Evolution')
        axes[0,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        # Price Positioning
        price_columns = [col for col in competitive_data.columns if 'Price' in col and col != 'Our Price (Lira)']
        if price_columns:
            competitive_data[['Our Price (Lira)'] + price_columns].plot(ax=axes[0,1], title='Price Positioning (Lira)')
            axes[0,1].set_ylabel('Price (Lira)')
        
        # Market Concentration (HHI)
        if 'HHI' in competitive_data.columns:
            competitive_data['HHI'].plot(ax=axes[1,0], color='purple', title='Market Concentration (HHI)')
            axes[1,0].set_ylabel('HHI Index')
        
        # Price Index
        if 'Price Index' in competitive_data.columns:
            competitive_data['Price Index'].plot(ax=axes[1,1], color='red', title='Relative Price Index')
            axes[1,1].axhline(y=1, color='black', linestyle='--')
            axes[1,1].set_ylabel('Price Index')
        
        plt.tight_layout()
        plt.show()

    def plot_market_expansion(self):
        """Plots market expansion metrics such as market penetration, patient growth, and revenue per patient"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Market Penetration
        self.sales_data['Segment Share'].plot(ax=axes[0,0], marker='o', color='blue')
        axes[0,0].set_title('Market Penetration')
        axes[0,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

        # Patient Growth
        self.sales_data['Patients on Product'].plot(ax=axes[0,1], marker='o', color='green')
        axes[0,1].set_title('Patient Growth')
        axes[0,1].set_ylabel('Number of Patients')

        # Revenue Growth
        self.sales_data['Revenue Growth'].plot(ax=axes[1,0], kind='bar', color='purple')
        axes[1,0].set_title('Revenue Growth')
        axes[1,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

        # Revenue per Patient
        self.sales_data['Revenue per Patient'].plot(ax=axes[1,1], marker='o', color='red')
        axes[1,1].set_title('Revenue per Patient (Lira)')
        axes[1,1].set_ylabel('Revenue (Lira)')
        
        plt.tight_layout()
        plt.show()

    def plot_patient_dynamics(self):
        """Plots patient dynamics including treatment rates and acquisition costs"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Patient Pool vs Treated Patients
        pd.DataFrame({
            'Total Patients': self.patient_data,
            'Treated Patients': self.sales_data['Patients on Product']
        }).plot(ax=axes[0,0])
        axes[0,0].set_title('Patient Pool vs Treated Patients')
        axes[0,0].set_ylabel('Number of Patients')

        # Treatment Rate
        treatment_rate = np.where(
            self.patient_data > 0,
            self.sales_data['Patients on Product'] / self.patient_data,
            0
        )
        pd.Series(treatment_rate, index=self.patient_data.index).plot(ax=axes[0,1], marker='o', color='green')
        axes[0,1].set_title('Treatment Rate')
        axes[0,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

        # Patient Acquisition Cost
        promo_cost_yearly = (1 + self.config.INFLATION) ** self.years * self.config.PROMO_COST_PER_PATIENT
        promo_spend_series = self.sales_data['Patients on Product'] * promo_cost_yearly
        acquisition_cost = np.where(
            self.sales_data['Patients on Product'] > 0,
            promo_spend_series / self.sales_data['Patients on Product'],
            0
        )
        pd.Series(acquisition_cost, index=self.sales_data.index).plot(ax=axes[1,0], marker='o', color='red')
        axes[1,0].set_title('Patient Acquisition Cost (Lira per Patient)')
        axes[1,0].set_ylabel('Cost per Patient (Lira)')

        # Cumulative New Patients
        new_patients = self.sales_data['Patients on Product'].diff().fillna(
            self.sales_data['Patients on Product'].iloc[0]
        )
        cumulative_new_patients = new_patients.cumsum()
        cumulative_new_patients.plot(ax=axes[1,1], color='purple')
        axes[1,1].set_title('Cumulative New Patients')
        axes[1,1].set_ylabel('Number of Patients')

        plt.tight_layout()
        plt.show()

    def plot_regional_analysis(self):
        """Plots regional market analysis including patient pools, penetration rates, and revenue potential"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Regional Patient Pool
        self.results['regional_data']['Patient Pool'].plot(kind='bar', ax=axes[0,0], color='blue')
        axes[0,0].set_title('Regional Market Size (Patients)')
        axes[0,0].set_ylabel('Number of Patients')

        # Regional Penetration Rate
        self.results['regional_data']['Penetration Rate'].plot(kind='bar', ax=axes[0,1], color='green')
        axes[0,1].set_title('Regional Penetration Rate')
        axes[0,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

        # Regional Revenue Potential
        self.results['regional_data']['Revenue Potential (Lira)'].plot(kind='bar', ax=axes[1,0], color='purple')
        axes[1,0].set_title('Regional Revenue Potential (Lira)')
        axes[1,0].set_ylabel('Lira')

        # Regional Growth Opportunity Index
        self.results['regional_data']['Growth Index'].plot(kind='bar', ax=axes[1,1], color='red')
        axes[1,1].set_title('Growth Opportunity Index')
        
        plt.tight_layout()
        plt.show()

    def plot_risk_analysis(self):
        """Plots risk analysis including NPV distribution, sensitivity correlations, Value at Risk, and scenario analysis"""
        monte_carlo = self.results['monte_carlo_results']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # NPV Distribution
        sns.histplot(monte_carlo['npv_results'], kde=True, ax=axes[0,0], color='blue')
        axes[0,0].axvline(monte_carlo['statistics']['mean'], color='red', linestyle='--',
                           label=f"Mean: {monte_carlo['statistics']['mean']:.1f}M Lira")
        axes[0,0].set_title('NPV Distribution (Millions of Lira)')
        axes[0,0].set_xlabel('NPV (Millions of Lira)')
        axes[0,0].legend()

        # Sensitivity Correlation
        correlations = monte_carlo['param_values'].corrwith(pd.Series(monte_carlo['npv_results'])).sort_values()
        correlations.plot(kind='barh', ax=axes[0,1], color='green')
        axes[0,1].set_title('Sensitivity Analysis')
        axes[0,1].set_xlabel('Correlation with NPV')

        # Value at Risk
        sorted_npv = np.sort(monte_carlo['npv_results'])
        probabilities = np.arange(1, len(sorted_npv) + 1) / len(sorted_npv)
        axes[1,0].plot(sorted_npv, probabilities, color='purple')
        axes[1,0].set_title('Value at Risk')
        axes[1,0].set_xlabel('NPV (Millions of Lira)')
        axes[1,0].set_ylabel('Cumulative Probability')

        # Scenario Analysis
        scenarios = pd.Series({
            'Worst Case': monte_carlo['statistics']['percentile_5'],
            'Base Case': monte_carlo['statistics']['median'],
            'Best Case': monte_carlo['statistics']['percentile_95']
        })
        scenarios.plot(kind='bar', ax=axes[1,1], color=['red', 'blue', 'green'])
        axes[1,1].set_title('Scenario Analysis')
        axes[1,1].set_ylabel('NPV (Millions of Lira)')

        plt.tight_layout()
        plt.show()

class ResultsExporter:
    """Exports model results to Excel and generates summary reports"""
    def __init__(self, results, config):
        self.results = results
        self.config = config

    def export_to_excel(self, filename='model_results_lira.xlsx'):
        """Exports various dataframes to an Excel file with multiple sheets"""
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            self.results['sales_data'].to_excel(writer, sheet_name='Sales Data')
            self.results['financials'].to_excel(writer, sheet_name='Financials')
            self.results['cash_flows'].to_excel(writer, sheet_name='Cash Flows')
            self.results['regional_data'].to_excel(writer, sheet_name='Regional Analysis')
            self.results['competitive_data'].to_excel(writer, sheet_name='Competitive Analysis')
            pd.DataFrame({'NPV Results (Millions of Lira)': self.results['monte_carlo_results']['npv_results']}).to_excel(writer, sheet_name='Monte Carlo')
            self.results['sensitivity_results'].to_excel(writer, sheet_name='Sensitivity')
        print(f"Results successfully exported to {filename}")

    def generate_summary_report(self):
        """Generates a summary report of key metrics"""
        summary = {
            'Market Metrics': {
                'Final Market Share': self.results['sales_data']['Segment Share'].iloc[-1],
                'Total Patients': self.results['sales_data']['Patients on Product'].iloc[-1],
                'Revenue (Lira)': self.results['sales_data']['Sales'].iloc[-1]
            },
            'Financial Metrics': {
                'NPV (Millions of Lira)': self.results['npv'],
                'IRR': self.results['irr'],
                'Payback Period': self.results['payback_period'],
                'Final Year EBITDA Margin': self.results['financials']['EBITDA Margin'].iloc[-1],  # Corrected Access
                'Average ROIC': self.results['summary_metrics']['avg_roic']
            },
            'Risk Metrics': {
                'NPV Mean (Millions of Lira)': self.results['monte_carlo_results']['statistics']['mean'],
                'NPV Std Dev (Millions of Lira)': self.results['monte_carlo_results']['statistics']['std'],
                'Probability of Negative NPV': self.results['monte_carlo_results']['statistics']['prob_negative']
            }
        }
        summary_df = pd.DataFrame(summary)
        return summary_df

def main():
    try:
        config = ModelConfig()
        print("Initializing financial model in Lira...")
        model = FinancialModel(config)
        
        print("Running analysis...")
        results = model.run_full_analysis()
        
        print("Generating visualizations...")
        visualizer = ModelVisualizer(results, config)
        visualizer.create_all_visualizations()
        
        print("Exporting results in Lira...")
        exporter = ResultsExporter(results, config)
        exporter.export_to_excel()
        summary = exporter.generate_summary_report()
        
        print("\nAnalysis Complete!")
        print("\nKey Metrics Summary (in Lira):")
        print(summary)
        
        return results
        
    except Exception as e:
        print("An error occurred during the financial model execution.")
        print(str(e))
        raise e

if __name__ == "__main__":
    results = main()
