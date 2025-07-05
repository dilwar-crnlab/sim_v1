#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean GSNR Calculator Implementation
Based on the exact equations from the research papers
Follows Steps 1-7 as described in GSNR.docx exactly

No code reuse from steps folder - clean implementation from scratch
Uses config.py for all physical parameters
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from dataclasses import dataclass

from config import MCF4CoreCLBandConfig

@dataclass
class GSNRResult:
    """Clean GSNR calculation result"""
    channel_index: int
    core_index: int
    gsnr_db: float
    osnr_db: float
    supported_modulation: str
    max_bitrate_gbps: int
    
    # Noise components
    ase_power_w: float
    nli_power_w: float
    icxt_power_w: float
    
    # SNR components
    snr_ase_db: float
    snr_nli_db: float
    snr_icxt_db: float
    
    # Additional metrics
    path_length_km: float
    num_interfering_channels: int
    calculation_time_s: float

class CleanGSNRCalculator:
    """
    Clean GSNR Calculator following the exact paper methodology
    Steps 1-7 implementation without legacy code dependencies
    """
    
    def __init__(self, mcf_config):
        """
        Initialize with MCF configuration from config.py
        
        Args:
            mcf_config: MCF4CoreCLBandConfig instance
        """
        self.config = mcf_config
        
        # Extract essential parameters
        self.channels = mcf_config.channels
        self.num_channels = len(self.channels)
        self.frequencies_hz = np.array([ch['frequency_hz'] for ch in self.channels])
        self.wavelengths_nm = np.array([ch['wavelength_nm'] for ch in self.channels])
        
        # Physical constants from config
        self.h_planck = mcf_config.physical_constants['h_planck']
        self.c_light = mcf_config.physical_constants['c_light']
        self.symbol_rate = mcf_config.physical_constants['symbol_rate_hz']
        
        # MCF parameters
        self.mcf_params = mcf_config.mcf_params
        self.freq_params = mcf_config.frequency_dependent_params
        
        # Modulation format thresholds and bitrates
        self.mod_thresholds = mcf_config.get_modulation_thresholds_db()
        self.mod_bitrates = mcf_config.get_modulation_bitrates_gbps()
        
        print(f"Clean GSNR Calculator initialized:")
        print(f"  Channels: {self.num_channels}")
        print(f"  MCF: {self.mcf_params.num_cores} cores, {self.mcf_params.core_pitch_um}μm pitch")
        print(f"  Using exact paper methodology for Steps 1-7")
    
    def calculate_gsnr(self, path_links: List, channel_index: int, core_index: int,
                  launch_power_dbm: float = 0.0, 
                  interfering_channels: List[int] = None,
                  link_wise_interference: Dict[int, List[int]] = None,
                  spectrum_allocation=None) -> GSNRResult:
        """
        Complete span-by-span GSNR calculation with backward compatibility
        """
        start_time = time.time()
        
        # Handle backward compatibility - convert interfering_channels to link_wise_interference
        if link_wise_interference is None and interfering_channels is not None:
            # Create link_wise_interference from legacy interfering_channels
            link_wise_interference = {}
            for link in path_links:
                link_wise_interference[link.link_id] = interfering_channels
        
        # Get link-wise interference if not provided
        if link_wise_interference is None:
            if spectrum_allocation is not None:
                link_wise_interference = spectrum_allocation.get_link_wise_interfering_channels(
                    [link.link_id for link in path_links], core_index, channel_index
                )
            else:
                link_wise_interference = {}
        
        # Initialize span-wise SNR sums per paper Equations (5-7)
        snr_ase_sum = 0.0
        snr_nli_sum = 0.0
        snr_icxt_sum = 0.0

        current_span_launch_power_w = 10**(launch_power_dbm/10) * 1e-3
        span_counter = 0
        
        # Process each span using existing step methods with link-specific interference
        for link in path_links:
            # Get interfering channels specific to this link
            link_interfering_channels = link_wise_interference.get(link.link_id, [])
            
            for span_idx in range(link.num_spans):
                span_length_km = link.span_lengths_km[span_idx]
                
                # Create single-span link list for existing methods
                single_span_links = [type('SpanLink', (), {
                    'length_km': span_length_km,
                    'link_id': link.link_id
                })()]

                single_span_path = [type('SpanLink', (), {
                    'length_km': span_length_km,
                    'link_id': link.link_id,
                    'num_spans': 1,
                    'span_lengths_km': [span_length_km]
                })()]

                # CALCULATE ICXT FOR THIS SPAN ONLY
                span_icxt_result = self._calculate_icxt(
                    single_span_path,  # Single span, not entire path!
                    channel_index, 
                    core_index, 
                    spectrum_allocation, 
                    10 * np.log10(current_span_launch_power_w * 1000)
                )
                
                # Step 1: Power evolution for this span starting from current power level
                span_step1_result = self._step1_power_evolution(
                    single_span_links, channel_index, link_interfering_channels, 
                    10 * np.log10(current_span_launch_power_w * 1000)  # Convert to dBm
                )
                
                # Step 2: Use existing parameter fitting for this span
                span_step2_result = self._step2_parameter_fitting(span_step1_result, channel_index)
                
                # Step 3: Use existing ASE calculation for this span
                span_step3_result = self._step3_ase_calculation(span_step1_result, span_step2_result, channel_index)
                
                # Step 4: Use existing parameter M calculation
                span_step4_result = self._step4_parameter_m(span_step2_result, channel_index)
                
                # Step 5: Use existing dispersion profile calculation with link-specific interference
                span_step5_result = self._step5_dispersion_profile(channel_index, link_interfering_channels)
                
                # Step 6: Use existing nonlinearity coefficient calculation with link-specific interference
                span_step6_result = self._step6_nonlinearity_coefficient(channel_index, link_interfering_channels)
                
                # Step 7: Use existing NLI calculation for this span with link-specific interference
                span_step7_result = self._step7_nli_calculation(
                    span_step1_result, span_step2_result, span_step4_result, 
                    span_step5_result, span_step6_result, channel_index, link_interfering_channels
                )
                
                # Use existing ICXT calculation for this span
                #span_icxt_result = self._calculate_icxt(single_span_links, channel_index, core_index)
                
                # P^s+1,i_tx: Launch power at span input (restored by amplifier)
                #P_tx_span = 10**(launch_power_dbm/10) * 1e-3
                # P^s+1,i_tx: ACTUAL launch power for this span
                P_tx_span = current_span_launch_power_w
                
                # P^s,i_ASE, P^s,i_NLI, P^s,i_ICXT: Span noise powers
                P_ase_span = span_step3_result['ase_power_w']
                P_nli_span = span_step7_result['nli_power_w'] 
                P_icxt_span = span_icxt_result['icxt_power_w']
                
                # Add span contributions per paper Equations (5-7)
                snr_ase_sum += P_tx_span / max(P_ase_span, 1e-15)
                snr_nli_sum += P_tx_span / max(P_nli_span, 1e-15)
                snr_icxt_sum += P_tx_span / max(P_icxt_span, 1e-15)

                # Update power for next span (amplifier restores to launch level)
                span_output_power = span_step1_result['final_powers_w'][channel_index]
                if span_output_power > 1e-15:
                    # Amplifier gain restores power
                    amplifier_gain = current_span_launch_power_w / span_output_power
                    current_span_launch_power_w = span_output_power * amplifier_gain
                else:
                    current_span_launch_power_w = 10**(launch_power_dbm/10) * 1e-3  # Reset to launch
                
                span_counter += 1
        
        # Transceiver SNR
        snr_trx_linear = 10**(30.0 / 10)
        
        # Final GSNR calculation per paper Equation (4)
        epsilon = 1e-12
        combined_inv_snr = (1/max(snr_ase_sum, epsilon) + 1/max(snr_nli_sum, epsilon) + 
                            1/max(snr_icxt_sum, epsilon) + 1/snr_trx_linear)
        gsnr_linear = max(1 / combined_inv_snr, epsilon)
        
        # Convert to dB with system penalties
        filtering_penalty_db = 1.0
        aging_margin_db = 1.0
        gsnr_db = 10 * np.log10(gsnr_linear) - filtering_penalty_db - aging_margin_db
        
        # Calculate OSNR
        osnr_db = gsnr_db + 3.0
        
        # Determine supported modulation format using existing config method
        supported_modulation, max_bitrate = self.config.get_supported_modulation_format(gsnr_db)
        
        # Path length
        path_length_km = sum(link.length_km for link in path_links)
        
        # Total interfering channels across all links
        total_interfering = set()
        for channels in link_wise_interference.values():
            total_interfering.update(channels)
        
        return GSNRResult(
            channel_index=channel_index,
            core_index=core_index,
            gsnr_db=gsnr_db,
            osnr_db=osnr_db,
            supported_modulation=supported_modulation,
            max_bitrate_gbps=max_bitrate,
            ase_power_w=0,  # Not applicable for span-wise calculation
            nli_power_w=0,  # Not applicable for span-wise calculation
            icxt_power_w=0, # Not applicable for span-wise calculation
            snr_ase_db=10 * np.log10(max(snr_ase_sum, epsilon)),
            snr_nli_db=10 * np.log10(max(snr_nli_sum, epsilon)),
            snr_icxt_db=10 * np.log10(max(snr_icxt_sum, epsilon)),
            path_length_km=path_length_km,
            num_interfering_channels=len(total_interfering),
            calculation_time_s=time.time() - start_time
        )
    
    def _step1_power_evolution(self, path_links: List, channel_index: int, 
                              interfering_channels: List[int], launch_power_dbm: float) -> Dict:
        """
        Step 1: Exact Power Evolution Profile calculation
        Implements Equation (10): ∂P/∂z = κP[Σ ζ(fi/fj) × Cr(fj, fj-fi)P(fj,z) - α(fi)]
        """
        #print("Called")
        # Calculate total path length
        total_length_km = sum(link.length_km for link in path_links)
        total_length_m = total_length_km * 1000
        
        # Active channels (target + interferers)
        active_channels = [channel_index] + interfering_channels
        
        # Initial power setup
        launch_power_w = 10**(launch_power_dbm/10) * 1e-3
        initial_powers = np.zeros(self.num_channels)
        initial_powers[channel_index] = launch_power_w
        
        # Set interfering channel powers (assume same level)
        for ch_idx in interfering_channels:
            initial_powers[ch_idx] = launch_power_w
        
        # Distance array for integration
        num_points = max(1000, int(total_length_m / 1000))  # 100m resolution minimum
        distances_m = np.linspace(0, total_length_m, num_points)
        
        # Solve coupled differential equations
        def power_derivatives(z, powers):
            """Calculate dP/dz for all channels"""
            dpdt = np.zeros_like(powers)
            
            for i in active_channels:
                if powers[i] <= 0:
                    continue
                
                # Get frequency-dependent loss
                freq_i = self.frequencies_hz[i]
                alpha_i = self.freq_params['loss_coefficient_db_km'][freq_i] * np.log(10) / (10 * 1000)
                
                # Raman interaction sum
                raman_sum = 0.0
                for j in active_channels:
                    if i != j and powers[j] > 0:
                        freq_j = self.frequencies_hz[j]
                        
                        # ζ function: frequency ratio filter
                        freq_ratio = freq_i / freq_j
                        if freq_ratio > 1:
                            zeta = freq_ratio
                        elif freq_ratio == 0:
                            zeta = 0
                        else:
                            zeta = 1
                        
                        # Raman gain coefficient
                        freq_diff_thz = abs(freq_j - freq_i) / 1e12
                        cr_gain = self._calculate_raman_gain(freq_diff_thz)
                        
                        raman_sum += zeta * cr_gain * powers[j]
                
                # Apply Equation (10)
                kappa = 1.0  # Forward propagation
                dpdt[i] = kappa * powers[i] * (raman_sum - alpha_i)
            
            return dpdt
        
        # Solve ODE system
        sol = solve_ivp(
            power_derivatives, 
            [0, total_length_m], 
            initial_powers,
            t_eval=distances_m,
            method='RK45',
            rtol=1e-8
        )
        
        power_evolution = sol.y.T  # Transpose to get (distance, channel) shape
        final_powers = power_evolution[-1, :]
        
        return {
            'distances_m': distances_m,
            'distances_km': distances_m / 1000,
            'power_evolution_w': power_evolution,
            'initial_powers_w': initial_powers,
            'final_powers_w': final_powers,
            'total_length_km': total_length_km,
            'active_channels': active_channels,
            'target_channel': channel_index
        }
    
    def _step2_parameter_fitting(self, step1_result: Dict, channel_index: int) -> Dict:
        """
        Step 2: Fit auxiliary loss coefficients α₀(f), α₁(f), σ(f)
        Using the approximate model and cost function from the paper
        """
        
        # Extract power evolution for target channel
        P_num = step1_result['power_evolution_w'][:, channel_index]
        P0 = P_num[0]
        z = step1_result['distances_m']
        
        if P0 <= 1e-15:
            # Return fallback parameters for inactive channels
            freq_hz = self.frequencies_hz[channel_index]
            alpha_intrinsic = self.freq_params['loss_coefficient_db_km'][freq_hz] * np.log(10) / (10 * 1000)
            return {
                'alpha0_per_m': alpha_intrinsic,
                'alpha1_per_m': 0.0,
                'sigma_per_m': 2 * alpha_intrinsic,
                'cost_value': 1e10,
                'r_squared': 0.0
            }
        
        # Get intrinsic loss for this frequency
        freq_hz = self.frequencies_hz[channel_index]
        alpha_intrinsic = self.freq_params['loss_coefficient_db_km'][freq_hz] * np.log(10) / (10 * 1000)
        
        def approximate_model(z_arr, alpha0, alpha1, sigma):
            """Approximate power model from Equation (13)"""
            if sigma <= 0:
                return P0 * np.exp(-2 * alpha0 * z_arr)
            
            exponent = -2 * alpha0 * z_arr + (2 * alpha1 / sigma) * (np.exp(-sigma * z_arr) - 1)
            return P0 * np.exp(exponent)
        
        def cost_function(sigma):
            """Cost function for σ optimization"""
            try:
                # Solve for α₀ and α₁ given σ (closed-form solution)
                alpha0, alpha1 = self._solve_alpha0_alpha1(sigma, P_num, P0, z)
                
                # Calculate cost using Equation (24)
                P_approx = approximate_model(z, alpha0, alpha1, sigma)
                #rel_error = np.log(P_num / P0) + 2 * alpha0 * z + 2 * alpha1 * (1 - np.exp(-sigma * z)) / sigma

                epsilon = 1e-15
                P_num_safe = np.maximum(P_num, epsilon)
                P0_safe = max(P0, epsilon)
                rel_error = np.log(P_num_safe / P0_safe) + 2 * alpha0 * z + 2 * alpha1 * (1 - np.exp(-sigma * z)) / sigma
                
                # Weight by power^2 and integrate
                weight = P_num**2
                cost = np.trapz(weight * rel_error**2, z)
                
                return cost
            except:
                return 1e10
        
        # Optimize σ in valid range
        sigma_min = alpha_intrinsic
        sigma_max = 4 * alpha_intrinsic
        
        result = minimize_scalar(cost_function, bounds=(sigma_min, sigma_max), method='bounded')
        optimal_sigma = result.x
        
        # Calculate final α₀ and α₁
        alpha0, alpha1 = self._solve_alpha0_alpha1(optimal_sigma, P_num, P0, z)
        
        # Calculate goodness of fit
        P_fitted = approximate_model(z, alpha0, alpha1, optimal_sigma)
        ss_res = np.sum((P_num - P_fitted)**2)
        ss_tot = np.sum((P_num - np.mean(P_num))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'alpha0_per_m': alpha0,
            'alpha1_per_m': alpha1, 
            'sigma_per_m': optimal_sigma,
            'cost_value': result.fun,
            'r_squared': r_squared
        }
    
    def _solve_alpha0_alpha1(self, sigma: float, P_num: np.ndarray, P0: float, z: np.ndarray) -> Tuple[float, float]:
        """
        Solve for α₀ and α₁ given σ using equations (30.1) and (30.2)
        Closed-form solution from the paper
        """
        try:
            # Calculate matrix elements
            exp_neg_sigma_z = np.exp(-sigma * z)
            one_minus_exp = (1 - exp_neg_sigma_z) / sigma
            
            # Weight function (power^2)
            weight = P_num**2
            
            # Matrix equation: A * [α₀, α₁]ᵀ = b
            A11 = np.trapz(weight * z**2, z)
            A12 = np.trapz(weight * z * one_minus_exp, z)
            A22 = np.trapz(weight * one_minus_exp**2, z)

            # Add numerical stability to avoid log(0) or log(negative)
            epsilon = 1e-15
            P_num_safe = np.maximum(P_num, epsilon)
            P0_safe = max(P0, epsilon)
            log_ratio = np.log(P_num_safe / P0_safe)
            b1 = -0.5 * np.trapz(weight * z * log_ratio, z)
            b2 = -0.5 * np.trapz(weight * one_minus_exp * log_ratio, z)
            
            # Solve 2x2 system
            det = A11 * A22 - A12**2
            if abs(det) < 1e-12:
                # Fallback to intrinsic loss
                freq_hz = self.frequencies_hz[0]  # Use first channel as reference
                alpha0 = self.freq_params['loss_coefficient_db_km'][freq_hz] * np.log(10) / (10 * 1000)
                alpha1 = 0.0
            else:
                alpha0 = (A22 * b1 - A12 * b2) / det
                alpha1 = (A11 * b2 - A12 * b1) / det
            
            return alpha0, alpha1
            
        except:
            # Fallback values
            freq_hz = self.frequencies_hz[0]
            alpha0 = self.freq_params['loss_coefficient_db_km'][freq_hz] * np.log(10) / (10 * 1000)
            alpha1 = 0.0
            return alpha0, alpha1
    
    def _step3_ase_calculation(self, step1_result: Dict, step2_result: Dict, channel_index: int) -> Dict:
        """
        Step 3: Calculate amplifier gain and ASE noise power
        Using Equation (5): G = P_out / P_in and ASE formula
        """
        
        # Get channel band for noise figure
        wavelength_nm = self.wavelengths_nm[channel_index]
        if wavelength_nm < 1565:
            noise_figure_db = self.config.amplifier_config.c_band_noise_figure_db
        else:
            noise_figure_db = self.config.amplifier_config.l_band_noise_figure_db
        
        noise_figure_linear = 10**(noise_figure_db / 10)
        
        # Calculate amplifier gain
        P_in = step1_result['final_powers_w'][channel_index]
        P_target = step1_result['initial_powers_w'][channel_index]  # Restore to launch power
        
        if P_in > 0:
            gain_linear = P_target / P_in
            gain_db = 10 * np.log10(gain_linear)
        else:
            gain_linear = 100  # 20 dB default
            gain_db = 20.0
        
        # Calculate ASE noise power
        freq_hz = self.frequencies_hz[channel_index]
        P_ase = noise_figure_linear * self.h_planck * freq_hz * (gain_linear - 1) * self.symbol_rate
        
        return {
            'gain_linear': gain_linear,
            'gain_db': gain_db,
            'noise_figure_db': noise_figure_db,
            'ase_power_w': P_ase
        }
    
    def _step4_parameter_m(self, step2_result: Dict, channel_index: int) -> Dict:
        """
        Step 4: Calculate parameter M
        M = floor(10 × |2α₁(f)/σ(f)|) + 1
        """
        
        alpha1 = step2_result['alpha1_per_m']
        sigma = step2_result['sigma_per_m']
        
        if sigma > 0:
            ratio = abs(2 * alpha1 / sigma)
            M = int(np.floor(10 * ratio)) + 1
        else:
            M = 1
        
        # Ensure reasonable bounds
        M = max(1, min(M, 20))
        
        return {'M': M}
    
    def _step5_dispersion_profile(self, channel_index: int, interfering_channels: List[int]) -> Dict:
        """
        Step 5: Calculate effective dispersion profile β₂(f)
        Using frequency-dependent dispersion coefficients
        """
        
        # Dispersion parameters from config
        beta2_ref = -21.86e-27  # ps²/m at reference frequency
        beta3 = 0.1331e-39      # ps³/m
        beta4 = -2.7e-55        # ps⁴/m
        f0 = 193.4e12           # Reference frequency
        
        # Calculate β₂ for each channel pair interaction
        beta2_eff = {}
        
        all_channels = [channel_index] + interfering_channels
        
        for i in all_channels:
            for j in all_channels:
                fi = self.frequencies_hz[i]
                fj = self.frequencies_hz[j]
                
                # Equation (6) from paper
                beta2_ij = (beta2_ref + 
                           np.pi * beta3 * (fi + fj - 2 * f0) +
                           (np.pi**2 / 3) * beta4 * 
                           ((fi - f0)**2 + (fi - f0) * (fj - f0) + (fj - f0)**2))
                
                beta2_eff[(i, j)] = beta2_ij
        
        return {'beta2_effective': beta2_eff}
    
    def _step6_nonlinearity_coefficient(self, channel_index: int, interfering_channels: List[int]) -> Dict:
        """
        Step 6: Calculate frequency-dependent nonlinearity coefficient γᵢⱼ
        γᵢⱼ = (2πfᵢ/c) × (2n₂) / (Aₑff(fᵢ) + Aₑff(fⱼ))
        """
        
        n2 = 2.6e-20  # Nonlinear refractive index (m²/W)
        
        gamma_ij = {}
        all_channels = [channel_index] + interfering_channels
        
        for i in all_channels:
            for j in all_channels:
                fi = self.frequencies_hz[i]
                fj = self.frequencies_hz[j]
                
                # Get effective areas
                Aeff_i = self.freq_params['effective_area_um2'][fi] * 1e-12  # Convert to m²
                Aeff_j = self.freq_params['effective_area_um2'][fj] * 1e-12
                
                # Calculate γᵢⱼ
                gamma = (2 * np.pi * fi / self.c_light) * (2 * n2) / (Aeff_i + Aeff_j)
                gamma_ij[(i, j)] = gamma
        
        return {'gamma_ij': gamma_ij}
    
    def _step7_nli_calculation(self, step1_result: Dict, step2_result: Dict, 
                              step4_result: Dict, step5_result: Dict, step6_result: Dict,
                              channel_index: int, interfering_channels: List[int]) -> Dict:
        """
        Step 7: Calculate NLI noise power using enhanced GN model
        Complete implementation of Equation (6)
        """
        
        if not interfering_channels:
            return {'nli_power_w': 0.0}
        
        # Get signal power
        signal_power = step1_result['final_powers_w'][channel_index]
        
        if signal_power <= 0:
            return {'nli_power_w': 0.0}
        
        # Parameters
        M = step4_result['M']
        alpha0 = step2_result['alpha0_per_m']
        alpha1 = step2_result['alpha1_per_m']
        sigma = step2_result['sigma_per_m']
        
        nli_sum = 0.0
        
        # Sum over interfering channels
        for j_idx in interfering_channels:
            interferer_power = step1_result['final_powers_w'][j_idx]
            
            if interferer_power <= 0:
                continue
            
            # Get interaction parameters
            gamma_ij = step6_result['gamma_ij'][(channel_index, j_idx)]
            beta2_eff = step5_result['beta2_effective'][(channel_index, j_idx)]
            
            # Kronecker delta (0 for different channels)
            delta_ij = 0.0
            
            # Sum over polarizations and series terms
            for p in range(2):  # Polarizations
                for k in range(M):
                    for q in range(M):
                        
                        # Calculate ψ function (simplified for efficiency)
                        psi_val = self._calculate_psi_function(
                            channel_index, j_idx, p, k, alpha0, sigma, beta2_eff
                        )
                        
                        # ISRS correction factor
                        isrs_factor = np.exp(-4 * alpha1 / sigma) if sigma > 0 else 1.0
                        
                        # Machine learning correction term (set to 1 for exact calculation)
                        rho_j = 1.0
                        
                        # Series term with factorials
                        try:
                            k_factorial = np.math.factorial(k) if k < 170 else 1e100
                            q_factorial = np.math.factorial(q) if q < 170 else 1e100
                            
                            if k_factorial < 1e100 and q_factorial < 1e100:
                                denominator = (2 * np.pi * self.symbol_rate**2 * k_factorial * q_factorial *
                                             (4 * alpha0 + (k + q) * sigma) * abs(beta2_eff))
                                
                                if denominator > 0:
                                    nli_term = (rho_j * gamma_ij**2 * interferer_power**2 * 
                                               (2 - delta_ij) * ((-1)**p) * isrs_factor * psi_val / denominator)
                                    
                                    nli_sum += nli_term
                        except:
                            continue
        
        # Apply main coefficient (16/27 from Equation 6)
        P_nli = (16/27) * signal_power * abs(nli_sum)
        
        return {'nli_power_w': max(0, P_nli)}
    
    def _calculate_psi_function(self, i: int, j: int, p: int, k: int, 
                           alpha0_fj: float, alpha_fj: float, beta2_eff: float) -> float:
        """Calculate ψ function exactly per Equation (8)"""
        
        #print("Called")
        fi = self.frequencies_hz[i]
        fj = self.frequencies_hz[j]
        
        # Get channel-specific symbol rates
        #Rs_i = self.mcf_config.physical_constants['symbol_rate_hz']  # Signal channel symbol rate
        Rs_i = self.config.physical_constants['symbol_rate_hz']  # ✅ Correct
        #Rs_j = self.mcf_config.physical_constants['symbol_rate_hz']  # Interfering channel symbol rate
        Rs_j = self.config.physical_constants['symbol_rate_hz']  # ✅ Correct
        
       #print("p in psi", p)
        # Exact paper formula: (fj - fi + (-1)^p × Rs,j/2)
        freq_term = fj - fi + ((-1)**p) * (Rs_j / 2)
        
        # Numerator: π × 2 × β₂(fj) × Rs,i × freq_term
        numerator = np.pi ** 2 * beta2_eff * Rs_i * freq_term
        
        # Denominator: 2α₀(fj) + k × α(fj)
        denominator = 2 * alpha0_fj + k * alpha_fj
        
        if abs(denominator) > 1e-15:
            asinh_arg = numerator / denominator
            # Numerical stability for large arguments
            if abs(asinh_arg) > 100:
                psi = np.sign(asinh_arg) * np.log(2 * abs(asinh_arg))
            else:
                psi = np.arcsinh(asinh_arg)
        else:
            psi = 0.0
        
        return psi
    
    def _calculate_icxt_old(self, path_links: List, channel_index: int, core_index: int, 
                   spectrum_allocation=None, launch_power_dbm: float = 0.0) -> Dict:
        """Calculate ICXT using actual adjacent core powers from spectrum allocation"""
        
        # MCF parameters from config
        num_cores = self.config.mcf_params.num_cores
        core_pitch_m = self.config.mcf_params.core_pitch_um * 1e-6
        bending_radius_m = self.config.mcf_params.bending_radius_mm * 1e-3
        ncore = self.config.mcf_params.core_refractive_index
        
        # Path length
        total_length_m = sum(link.length_km for link in path_links) * 1000
        
        # Channel frequency
        freq_hz = self.frequencies_hz[channel_index]
        
        # Calculate mode coupling coefficient κ(f) and power coupling coefficient Ω(f)
        wavelength_m = self.c_light / freq_hz
        core_radius_m = self.config.mcf_params.core_radius_um * 1e-6
        delta = self.config.mcf_params.core_cladding_delta
        
        V1 = 2 * np.pi * core_radius_m * ncore * np.sqrt(2 * delta) / wavelength_m
        W1 = max(0.1, 1.143 * V1 - 0.22)
        
        kappa = (1e-6 / core_pitch_m) * (V1**2 / max(W1**3, 1e-10)) * np.exp(-W1)
        omega = (self.c_light * kappa**2 * bending_radius_m * ncore) / (np.pi * freq_hz * core_pitch_m)
        
        # Get adjacent cores for the current core
        adjacent_cores = self._get_adjacent_cores(core_index, num_cores)
        
        # Calculate actual ICXT power from spectrum allocation
        total_icxt_power = 0.0
        active_adjacent_cores = 0
        
        if spectrum_allocation is not None:
            # Check each adjacent core for actual activity
            for adj_core in adjacent_cores:
                # Check if this adjacent core has active channels on same channel frequency
                for link in path_links:
                    if spectrum_allocation.allocation[link.link_id, adj_core, channel_index] > 0:
                        # Adjacent core is active - use actual launch power
                        adjacent_power_w = 10**(launch_power_dbm/10) * 1e-3
                        
                        # Calculate μ_ICXT for this adjacent core
                        omega_L = omega * total_length_m
                        exp_term = np.exp(-2 * omega_L)  # Single adjacent core interaction
                        
                        if exp_term < 1e-10:
                            mu_icxt = 0.5
                        else:
                            mu_icxt = (1 - exp_term) / (1 + exp_term)
                        
                        # Add ICXT contribution from this adjacent core
                        total_icxt_power += mu_icxt * adjacent_power_w
                        active_adjacent_cores += 1
                        break  # Found activity on this core
        # else:
        #     # Fallback: assume typical scenario (for backward compatibility)
        #     NAC = len(adjacent_cores)
        #     if NAC > 0:
        #         omega_L = omega * total_length_m
        #         exp_term = np.exp(-(NAC + 1) * omega_L)
                
        #         if exp_term < 1e-10:
        #             mu_icxt = NAC / (NAC + 1)
        #         else:
        #             mu_icxt = (NAC - NAC * exp_term) / (1 + NAC * exp_term)
                
        #         # Conservative assumption: half of adjacent cores are active
        #         active_ratio = 0.5
        #         adjacent_power_w = 10**(launch_power_dbm/10) * 1e-3
        #         total_icxt_power = mu_icxt * NAC * active_ratio * adjacent_power_w
        
        return {
            'icxt_power_w': max(0, total_icxt_power),
            'mu_icxt': total_icxt_power / max(10**(launch_power_dbm/10) * 1e-3, 1e-15),
            'omega': omega,
            'kappa': kappa,
            'NAC': active_adjacent_cores
        }

    def _calculate_icxt(self, path_links: List, channel_index: int, core_index: int, 
                   spectrum_allocation=None, launch_power_dbm: float = 0.0) -> Dict:
    
        # if self._icxt_call_count[call_key] == 1:
        #     is_single_span = len(path_links) == 1 and path_links[0].num_spans == 1
        #     calculation_type = "SPAN-WISE" if is_single_span else "PATH-WISE (ERROR!)"
        #     print(f"\n🔍 ICXT Debug - Channel {channel_index}, Core {core_index} [{calculation_type}]:")
        #     print(f"   Length: {L/1000:.1f} km, Links: {len(path_links)}")
        
        #print(f"🔍 ICXT DEBUG CALLED - Ch{channel_index}, Core{core_index}")
        trench_params = self.config.mcf_params.get_trench_parameters_for_icxt()
        r_1 = trench_params['r_1_m']
        Lambda = trench_params['Lambda_m'] 
        w_tr = trench_params['w_tr_m']
        r_b = trench_params['r_b_m']
        n_core = trench_params['n_core']
        Delta_1 = trench_params['Delta_1']
        
        L = sum(link.length_km for link in path_links) * 1000
        f_i = self.frequencies_hz[channel_index]
        
        # Calculate V₁, W₁, U₁², K₁, Γ, κ, Ω, NAC, μ_ICXT
        V_1 = (2 * np.pi * f_i * r_1 * n_core * np.sqrt(2 * Delta_1)) / self.c_light
        W_1 = max(0.1, 1.143 * V_1 - 0.22)
        U_1_squared = ((2 * np.pi * f_i * r_1 * n_core) / self.c_light)**2 * (2 * Delta_1)
        K_1 = np.sqrt(np.pi / (2 * W_1)) * np.exp(-W_1)
        
        Gamma = W_1 / (W_1 + 1.2 * (1 + V_1) * (w_tr / Lambda))
        
        sqrt_Gamma = np.sqrt(max(Gamma, 1e-10))
        ratio_term = U_1_squared / (V_1**3 * K_1**2)
        geometric_term = np.sqrt(np.pi * r_1) / (W_1 * Lambda)
        exponent = -(W_1 * Lambda + 1.2 * (1 + V_1) * w_tr) / r_1
        exp_term = np.exp(exponent)
        
        kappa = sqrt_Gamma * ratio_term * geometric_term * exp_term
        Omega = (self.c_light * kappa**2 * r_b * n_core) / (np.pi * f_i * Lambda)
        
        # Get active adjacent cores
        adjacent_cores = self._get_adjacent_cores(core_index, self.config.mcf_params.num_cores)
        #print(f"   Adjacent cores to Core {core_index}: {adjacent_cores}")
        NAC = 0
        if spectrum_allocation:
            for adj_core in adjacent_cores:
                for link in path_links:
                    if spectrum_allocation.allocation[link.link_id, adj_core, channel_index] > 0:
                        NAC += 1
                        break
        else:
            NAC = len(adjacent_cores) // 2
        
        # Calculate μ_ICXT using Equation 1
        if NAC > 0:
            exp_arg = -(NAC + 1) * Omega * L
            if exp_arg < -50:
                exp_term = 0.0
            else:
                exp_term = np.exp(exp_arg)
            mu_icxt = (NAC - NAC * exp_term) / (1 + NAC * exp_term)
        else:
            mu_icxt = 0.0
        
        launch_power_w = 10**(launch_power_dbm/10) * 1e-3
        icxt_power_w = mu_icxt * launch_power_w
        
        return {
            'icxt_power_w': max(0, icxt_power_w),
            'mu_icxt': mu_icxt,
            'omega': Omega,
            'kappa': kappa,
            'NAC': NAC
        }
    
    def _calculate_icxt_(self, path_links: List, channel_index: int, core_index: int, 
                   spectrum_allocation=None, launch_power_dbm: float = 0.0) -> Dict:
    
        # Add call counter to prevent spam
        if not hasattr(self, '_icxt_call_count'):
            self._icxt_call_count = {}
        
        call_key = f"ch{channel_index}_core{core_index}"
        if call_key not in self._icxt_call_count:
            self._icxt_call_count[call_key] = 0
        
        self._icxt_call_count[call_key] += 1
        
        # Only show debug for first call
        if self._icxt_call_count[call_key] == 1:
            print(f"\n🔍 ICXT Debug - Channel {channel_index}, Core {core_index}:")
        elif self._icxt_call_count[call_key] <= 5:
            print(f"   🔍 ICXT call #{self._icxt_call_count[call_key]} for Ch{channel_index}, Core{core_index}")
        elif self._icxt_call_count[call_key] == 6:
            print(f"   ⚠️  ICXT called {self._icxt_call_count[call_key]}+ times for Ch{channel_index}, Core{core_index} - suppressing further debug")
        
        # Your existing ICXT calculation code...
        trench_params = self.config.mcf_params.get_trench_parameters_for_icxt()
        r_1 = trench_params['r_1_m']
        Lambda = trench_params['Lambda_m'] 
        w_tr = trench_params['w_tr_m']
        r_b = trench_params['r_b_m']
        n_core = trench_params['n_core']
        Delta_1 = trench_params['Delta_1']
        
        L = sum(link.length_km for link in path_links) * 1000
        f_i = self.frequencies_hz[channel_index]
        
        # Only show detailed debug on first call
        if self._icxt_call_count[call_key] == 1:
            print(f"   Path length: {L/1000:.1f} km, Frequency: {f_i/1e12:.2f} THz")
        
        # Calculate V₁, W₁, U₁², K₁, Γ, κ, Ω, NAC, μ_ICXT
        V_1 = (2 * np.pi * f_i * r_1 * n_core * np.sqrt(2 * Delta_1)) / self.c_light
        W_1 = max(0.1, 1.143 * V_1 - 0.22)
        U_1_squared = ((2 * np.pi * f_i * r_1 * n_core) / self.c_light)**2 * (2 * Delta_1)
        K_1 = np.sqrt(np.pi / (2 * W_1)) * np.exp(-W_1)
        
        Gamma = W_1 / (W_1 + 1.2 * (1 + V_1) * (w_tr / Lambda))
        
        sqrt_Gamma = np.sqrt(max(Gamma, 1e-10))
        ratio_term = U_1_squared / (V_1**3 * K_1**2)
        geometric_term = np.sqrt(np.pi * r_1) / (W_1 * Lambda)
        exponent = -(W_1 * Lambda + 1.2 * (1 + V_1) * w_tr) / r_1
        exp_term = np.exp(exponent)
        
        kappa = sqrt_Gamma * ratio_term * geometric_term * exp_term
        Omega = (self.c_light * kappa**2 * r_b * n_core) / (np.pi * f_i * Lambda)
        
        # Get active adjacent cores - THIS IS THE KEY PART TO DEBUG
        adjacent_cores = self._get_adjacent_cores(core_index, self.config.mcf_params.num_cores)
        
        NAC = 0
        if spectrum_allocation and self._icxt_call_count[call_key] == 1:
            print(f"   Adjacent cores to Core {core_index}: {adjacent_cores}")
            print(f"   Checking interference from adjacent cores:")
            
            for adj_core in adjacent_cores:
                is_active = False
                active_links = []
                
                # Check ALL links in network, not just path links
                for link_id in range(spectrum_allocation.num_links):
                    if spectrum_allocation.allocation[link_id, adj_core, channel_index] > 0:
                        active_links.append(link_id)
                        is_active = True
                
                print(f"      Core {adj_core}: {'ACTIVE' if is_active else 'INACTIVE'} on links {active_links}")
                
                if is_active:
                    NAC += 1
        elif spectrum_allocation:
            # Don't debug, just calculate NAC
            for adj_core in adjacent_cores:
                for link_id in range(spectrum_allocation.num_links):
                    if spectrum_allocation.allocation[link_id, adj_core, channel_index] > 0:
                        NAC += 1
                        break
        else:
            NAC = len(adjacent_cores) // 2
        
        # Calculate μ_ICXT using Equation 1
        if NAC > 0:
            exp_arg = -(NAC + 1) * Omega * L
            if exp_arg < -50:
                exp_term = 0.0
            else:
                exp_term = np.exp(exp_arg)
            mu_icxt = (NAC - NAC * exp_term) / (1 + NAC * exp_term)
        else:
            mu_icxt = 0.0
        
        launch_power_w = 10**(launch_power_dbm/10) * 1e-3
        icxt_power_w = mu_icxt * launch_power_w
        
        # Only show results on first call
        if self._icxt_call_count[call_key] == 1:
            print(f"   NAC: {NAC}, μ_ICXT: {mu_icxt:.2e}, ICXT Power: {icxt_power_w:.2e} W")
        
        return {
            'icxt_power_w': max(0, icxt_power_w),
            'mu_icxt': mu_icxt,
            'omega': Omega,
            'kappa': kappa,
            'NAC': NAC
        }

    def _get_adjacent_cores(self, core_index: int, num_cores: int) -> List[int]:
        """Get adjacent cores for given core in 4-core square layout"""
        if num_cores == 4:
            # 4-core square layout: 0-1, 0-3, 1-2, 2-3
            adjacency = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]}
            return adjacency.get(core_index, [])
        else:
            # Generic: adjacent cores (simplified)
            adjacent = []
            for i in range(num_cores):
                if i != core_index:
                    adjacent.append(i)
            return adjacent[:6]  # Limit to 6 adjacent cores max


    def _aggregate_gsnr_pathwise(self, step1_result: Dict, step3_result: Dict, step7_result: Dict,
                       icxt_result: Dict, channel_index: int, core_index: int,
                       path_links: List, num_interfering: int, calc_time: float) -> GSNRResult:
        """
        Aggregate all noise sources and calculate final GSNR
        Using Equation (4): GSNR = [SNR_ASE⁻¹ + SNR_NLI⁻¹ + SNR_ICXT⁻¹ + SNR_TRx⁻¹]⁻¹
        """
        
        # Signal power
        signal_power_w = step1_result['final_powers_w'][channel_index]
        
        # Noise powers
        ase_power_w = step3_result['ase_power_w']
        nli_power_w = step7_result['nli_power_w']
        icxt_power_w = icxt_result['icxt_power_w']
        
        # Transceiver SNR (dB to linear)
        snr_trx_db = 30.0  # Typical value
        snr_trx_linear = 10**(snr_trx_db / 10)
        
        # Calculate individual SNRs with numerical stability
        epsilon = 1e-12
        snr_ase = signal_power_w / (ase_power_w + epsilon)
        snr_nli = signal_power_w / (nli_power_w + epsilon)
        snr_icxt = signal_power_w / (icxt_power_w + epsilon)

        # Ensure all SNRs are positive
        snr_ase = max(snr_ase, epsilon)
        snr_nli = max(snr_nli, epsilon)
        snr_icxt = max(snr_icxt, epsilon)
        
        # Combined GSNR calculation
        combined_inv_snr = (1/snr_ase + 1/snr_nli + 1/snr_icxt + 1/snr_trx_linear)
        gsnr_linear = max(1 / combined_inv_snr, epsilon)
        
        # Convert to dB with system penalties
        filtering_penalty_db = 0
        aging_margin_db = 0
        gsnr_db = 10 * np.log10(gsnr_linear) - filtering_penalty_db - aging_margin_db
        
        # Calculate OSNR (simplified)
        osnr_db = gsnr_db + 3.0  # Typical conversion
        
        # Determine supported modulation format
        supported_modulation, max_bitrate = self.config.get_supported_modulation_format(gsnr_db)
        supported_modulation, max_bitrate = self.config.get_supported_modulation_format(gsnr_db)  # ✅ Correct
        
        # Path length
        path_length_km = sum(link.length_km for link in path_links)
        
        return GSNRResult(
            channel_index=channel_index,
            core_index=core_index,
            gsnr_db=gsnr_db,
            osnr_db=osnr_db,
            supported_modulation=supported_modulation,
            max_bitrate_gbps=max_bitrate,
            ase_power_w=ase_power_w,
            nli_power_w=nli_power_w,
            icxt_power_w=icxt_power_w,
            snr_ase_db=10 * np.log10(max(snr_ase, epsilon)),
            snr_nli_db=10 * np.log10(max(snr_nli, epsilon)),
            snr_icxt_db=10 * np.log10(max(snr_icxt, epsilon)),
            path_length_km=path_length_km,
            num_interfering_channels=num_interfering,
            calculation_time_s=calc_time
        )
    
    def _aggregate_gsnr(self, step1_result: Dict, step3_result: Dict, step7_result: Dict,
                   icxt_result: Dict, channel_index: int, core_index: int,
                   path_links: List, num_interfering: int, calc_time: float) -> GSNRResult:
        """
        Span-by-span GSNR calculation per paper Equations (5-7)
        SNRASE = Σs∈S P^s+1,i_tx / P^s,i_ASE
        """
        
        # Initialize span-wise SNR sums (Equations 5-7)
        snr_ase_sum = 0.0
        snr_nli_sum = 0.0
        snr_icxt_sum = 0.0
        epsilon = 1e-12
        
        # Calculate launch power per span
        launch_power_w = 10**(0.0/10) * 1e-3  # 0 dBm default
        
        # Iterate through each span (per paper methodology)
        total_spans = 0
        for link in path_links:
            for span_idx in range(link.num_spans):
                span_length_km = link.span_lengths_km[span_idx]
                
                # P^s+1,i_tx: Launch power at span input (after amplifier)
                P_tx_span = launch_power_w
                
                # P^s,i_ASE: Span ASE noise power (Equation 8)
                freq_hz = self.frequencies_hz[channel_index]
                wavelength_nm = self.wavelengths_nm[channel_index]
                noise_figure_db = 5.0 if wavelength_nm < 1565 else 5.5
                noise_figure_linear = 10**(noise_figure_db / 10)
                
                # Span loss and gain
                alpha_db_km = self.freq_params['loss_coefficient_db_km'][freq_hz]
                span_loss_db = alpha_db_km * span_length_km
                span_gain_linear = 10**(span_loss_db / 10)
                
                P_ase_span = (noise_figure_linear * self.h_planck * freq_hz * 
                            (span_gain_linear - 1) * self.symbol_rate)
                
                # P^s,i_NLI: Span NLI noise power (simplified per span)
                total_nli = step7_result['nli_power_w']
                P_nli_span = total_nli * span_length_km / sum(link.length_km for link in path_links)
                
                # P^s,i_ICXT: Span ICXT power
                total_icxt = icxt_result['icxt_power_w']
                P_icxt_span = total_icxt * span_length_km / sum(link.length_km for link in path_links)
                
                # Add span contributions (Equations 5-7)
                snr_ase_sum += P_tx_span / (P_ase_span + epsilon)
                snr_nli_sum += P_tx_span / (P_nli_span + epsilon)
                snr_icxt_sum += P_tx_span / (P_icxt_span + epsilon)
                
                total_spans += 1
        
        # Transceiver SNR
        snr_trx_linear = 10**(30.0 / 10)
        
        # Combined GSNR calculation (Equation 4)
        combined_inv_snr = (1/max(snr_ase_sum, epsilon) + 1/max(snr_nli_sum, epsilon) + 
                            1/max(snr_icxt_sum, epsilon) + 1/snr_trx_linear)
        gsnr_linear = max(1 / combined_inv_snr, epsilon)
        
        # Convert to dB
        gsnr_db = 10 * np.log10(gsnr_linear)
        osnr_db = gsnr_db + 3.0
        
        # Determine supported modulation format
        supported_modulation, max_bitrate = self.config.get_supported_modulation_format(gsnr_db)
        
        # Path length
        path_length_km = sum(link.length_km for link in path_links)
        
        return GSNRResult(
            channel_index=channel_index,
            core_index=core_index,
            gsnr_db=gsnr_db,
            osnr_db=osnr_db,
            supported_modulation=supported_modulation,
            max_bitrate_gbps=max_bitrate,
            ase_power_w=step3_result['ase_power_w'],
            nli_power_w=step7_result['nli_power_w'],
            icxt_power_w=icxt_result['icxt_power_w'],
            snr_ase_db=10 * np.log10(max(snr_ase_sum, epsilon)),
            snr_nli_db=10 * np.log10(max(snr_nli_sum, epsilon)),
            snr_icxt_db=10 * np.log10(max(snr_icxt_sum, epsilon)),
            path_length_km=path_length_km,
            num_interfering_channels=num_interfering,
            calculation_time_s=calc_time
        )


    def _calculate_raman_gain(self, freq_diff_thz: float) -> float:
        """
        Calculate Raman gain coefficient for frequency difference
        Simplified model of silica fiber Raman gain spectrum
        """
        
        # Raman gain peaks for silica fiber (THz)
        raman_peaks = [13.2, 15.8, 17.6]
        raman_amplitudes = [1.0, 0.4, 0.2]
        raman_widths = [2.5, 3.0, 3.5]
        
        total_gain = 0.0
        for peak, amplitude, width in zip(raman_peaks, raman_amplitudes, raman_widths):
            # Lorentzian lineshape
            gain_component = amplitude * (width/2)**2 / ((freq_diff_thz - peak)**2 + (width/2)**2)
            total_gain += gain_component
        
        # Scale to realistic values
        return total_gain * 0.65e-13