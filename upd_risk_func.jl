using Printf
using TickTock
using Base.Threads
using SpecialFunctions
using Distributions

const _norm = Normal()

function precalculation(S,
                        S_div,
                        theta1,
                        theta2, 
                        theta3, 
                        muliplier_p_line_s_l, 
                        muliplier_p_wave_s_l, 
                        muliplier_p_cover_s_l, 
                        ae_s, 
                        gamma_s, 
                        L_common, 
                        sum_d_arr, 
                        hi_s,
                        b1,
                        b2,
                        mu)

    ae_arr = zeros(S, 1)
    gamma_arr = zeros(S, 1)
    hi_arr = zeros(S, 1)
    zeta_mat = zeros(S, L_common)
    part_with_gamma_theta_1 = zeros(L_common, 1)
    part_with_gamma_theta_2 = zeros(L_common, 1)
    part_with_gamma_theta_3 = zeros(L_common, 1)
    p1_arr = zeros(L_common, 1)
    p2_arr = zeros(L_common, 1)
    function compute_expression(x, theta)
        numerator = gamma(x - 1 + (1 / theta))
        denominator = gamma(x) * gamma(1 / theta)
        return numerator / denominator
    end

    for k = 1:size(S_div , 1)
        for k1 = S_div[k,1]:S_div[k,2]
            #ae
            ae_arr[k1] = ae_s[k]
            
            #gamma
            gamma_arr[k1] = gamma_s[k]

            #hi
            hi_arr[k1] = hi_s
            
            values = k1 .- sum_d_arr[1:L_common]
            j = div.(k1 .- 1, 1000) .+ 1
            start = (j .- 1) .* 1000 .+ 1
            endd = (j .- 1) .* 1000 .+ 30
            zeta_mat[k1, 1:L_common] .= (start .<= values) .& (values .<= endd)
        end
    end
    p_s_l_line = muliplier_p_line_s_l * (1 .- zeta_mat)
    p_s_l_wave = muliplier_p_wave_s_l * (1 .- zeta_mat)
    p_s_l_cover = muliplier_p_cover_s_l * zeta_mat

    for l = 1:L_common
        part_with_gamma_theta_1[l] = compute_expression(l, theta1)
        part_with_gamma_theta_2[l] = compute_expression(l, theta2)
        part_with_gamma_theta_3[l] = compute_expression(l, theta3)
        p1_arr[l] = (atan(b1[1] + b1[2] * (L_common - l + 1) + b1[3] * mu)/pi)+0.5
        p2_arr[l] = cdf(_norm,  b2[1] + b2[2] * (L_common - l + 1) + b2[3] * mu)
    end

    return ae_arr, gamma_arr, hi_arr, p_s_l_line, p_s_l_wave, p_s_l_cover, part_with_gamma_theta_1, part_with_gamma_theta_2, part_with_gamma_theta_3, p1_arr, p2_arr
end

function g1(a1, xmax, v, w, L, Lw, mu, ae, gamma)
    ae_not_equal_0_indicator = Int(ae != 0)
    ae_equal_0_indicator = Int(ae == 0)
    gamma_more_0_indicator = Int(gamma > 0)
    gamma_less_0_indicator = Int(gamma < 0)
    exp1 = a1[1] + 
           a1[2] * ae * log(xmax) + 
           a1[3] * (1 - mu) * log(xmax) * log(v * 3.6)* min(0, gamma) + 
           a1[4] * (1 - mu) * log(xmax) + 
           a1[5] * (1 - mu)^2 * log(xmax) * log(v * 3.6) * min(0, gamma) + 
           a1[6] * mu + 
           a1[7] * gamma_more_0_indicator * mu * log(xmax) * log(v * 3.6)
    exp2 = a1[8] + 
           a1[9] * gamma_less_0_indicator * log(xmax) * log(v * 3.6) + 
           a1[10] * gamma_more_0_indicator * log(v * 3.6) * (log(xmax))^2
    result = ae_not_equal_0_indicator * exp(exp1) + 
                 ae_equal_0_indicator * exp(exp2)
    return result
end

function g2(a2, xmax, v, w, L, Lw, mu, ae, gamma)
    # gamma_more_0_indicator = Int(gamma > 0);
    # exp_val = a2[1] + a2[2] * mu^2 +
              # a2[3] * log(v * 3.6) * gamma_more_0_indicator +
              # a2[4] * log(xmax)
    # result = exp(exp_val)
    # return result
 
    exp_val = a2[1] + a2[2] * mu + 
              a2[3] * log(v * 3.6) + 
              a2[4] * log(xmax)
    result = exp(exp_val)
    return result
end

function g3(a3, xmax, v, w, L, Lw, mu, ae, gamma)
    exp_val = a3[1] + a3[2] * mu * log(v * 3.6) +
              a3[3] * mu * (log(xmax))^2 +
              a3[4] * log(v * 3.6) +
              a3[5] * log(v * 3.6) * log(xmax)
    result = exp(exp_val)
    return result
end

function q_func(L)
    result = zeros(Int, 3 * L ^ 2)
    for i = 1:3 * L ^ 2
        if i >= 1 && i <= L^2
            result[i] = floor((i - 1) / L) + 1
        elseif i >= L^2 + 1 && i <= 2 * L^2
            result[i] = floor((i - 1 - L^2) / L) + 1
        elseif i >= 2 * L^2 + 1 && i <= 3 * L^2
            result[i] = floor((i - 1 - 2 * L^2) / L) + 1
        else
            error("i out of range");
        end
    end
    return result
end

function f_func(L)
    result = zeros(Int, 3 * L ^ 2)
    for i = 1:3 * L ^ 2
        if i >= 1 && i <= L^2
            result[i] = i - trunc((i - 1) / L) * L
        elseif i >= L^2 + 1 && i <= 2 * L^2
            result[i] = i - L^2 - trunc((i - 1 - L^2) / L) * L
        elseif i >= 2 * L^2 + 1 && i <= 3 * L^2
            result[i] = i - 2 * L^2 - trunc((i - 1 - 2 * L^2) / L) * L
        else
            error("i out of range");
        end
    end
    return result
end

function step_func(L, q_arr, f_arr)
    result = zeros(Int, 3 * L ^ 2)
    for i = 1:3 * L ^ 2
        val = f_arr[i] + q_arr[i] - 1
        if val <= L
            result[i] = q_arr[i]
        end
    end
    return result
end

function p_s_l_k_vs_combined(s, l, k, v, theta, a, L, w, Lw, mu, ae, gamma, model_type, tmp_gamma)
    tmp_base = if model_type == :line
        theta * g1(a, L-l+1, v, w, L, Lw, mu, ae, gamma)
    elseif model_type == :wave
        theta * g2(a, L-l+1, v, w, L, Lw, mu, ae, gamma)
    elseif model_type == :cover
        theta * g3(a, L-l+1, v, w, L, Lw, mu, ae, gamma)
    end

    pow1 = -(k - 1 + (1 / theta))
    pow2 = k - 1

    result = tmp_gamma[k] * ((1 + tmp_base)^pow1) * (tmp_base^pow2)
    return result
end

function M_C(f, v, step, p_type, p1_arr, p2_arr)
	# common_part = 4.5e6 * ((2 / (1 + exp(-0.2 * v))) - 1)
	# common_part = 4.5e6 * (1 - (-(9 / 200) * v + 1) * exp(-0.005 * (v ^ 2)))
	# common_part = 4.5e6 * (((9/200) ^ 2) * (v ^ 2))
	# common_part = 4.5e6 * ((9/200) * v)
	# predexp = ((60/200) ^ 2) * (v ^ 2) + 2 * (60/200) * v + 2
	# common_part = 4.5e6 * (1 - ((predexp * exp(-(60/200) * v))/2))
	# common_part = 4.5e6 * ((2.358/(1 + exp(-0.15 * v))) - (2.358/2))
	# common_part = 4.5e6 * (v ^ 2) * (((3/100) * sqrt(2)) ^ 2)
	# common_part = 4.5e6 * (9/200) * v
	# express = (((60/200) ^ 2) * (v ^ 2) + 2 * (60 / 200) * v + 2 ) * exp((-60 / 200) * v)
	# common_part = 4.5e6 * (1.148 - 1.148 * (express / 2))
	# express = ((v ^ 2) * exp(-0.5 * (v - 19))) + 625
	# common_part = 4.5e6 * (9/5000) * (express / (1 + exp(-0.5 * (v - 19))))
    # 

    # express = 1 - (1 / (1 + exp(-0.5 * (v - 19))))
    # common_part = 4.5e6 * ((v ^ 2) * express + 625 * (1 / (1 + exp(-0.5 * (v - 19))))) * (13.5 / 6232.95)
	# common_part = 4.5e6 * ((9/200) * v)
	# common_part = 4.5e6 * ((3/15500) * 13.5 * (v ^ 2))
	# common_part = 4.5e6 * ((13.5/7.576) * ((1 + exp(-0.15 * v)) ^ -1) - 0.5)
	common_part = 4.5e6 * ((13.6/14.79) * (1 - (0.045 * (v ^ 2) + 0.3 * v + 1) * exp(-0.3 * v) ))
	    
    if p_type == :line
        return p1_arr[f] * 1e5 + common_part * step
    elseif p_type == :wave
        return p2_arr[f] * 1e5 + common_part * step
    else
        return 1e5 + common_part * step
    end

end

function SigmoidRiskFunction(v_var, S_div, L_common, L1, q_arr, f_arr, step_arr, ae_arr, gamma_arr, hi_arr, theta1, theta2, theta3, a1, a2, a3, p1_arr, p2_arr, p_s_l_line, p_s_l_wave, p_s_l_cover, P_s, part_with_gamma_theta_1, part_with_gamma_theta_2, part_with_gamma_theta_3, w, mu, sum_d_arr)
    s_values = Int[]
    for k = 1:size(S_div, 1)
        append!(s_values, S_div[k,1]:S_div[k,2])
    end

    summ_r1_ns = zeros(eltype(v_var), length(s_values))
    summ_r2_ns = zeros(eltype(v_var), length(s_values))

    @Threads.threads for idx = 1:length(s_values)
        s = s_values[idx]
        summ_r1_n = 0
        summ_r2_n = 0

        for i = 1:3*L_common^2
            q_i = q_arr[i]
            f_i = f_arr[i]
            step = step_arr[i]
            value = s - sum_d_arr[f_i]
            ae, gamma, hi = value < 1 ? (Float64(0), Float64(0), Float64(0)) : (ae_arr[value], gamma_arr[value], hi_arr[value])
            type_symbol = i <= L_common^2 ? :line : (i <= 2*L_common^2 ? :wave : :cover)
            theta = i <= L_common^2 ? theta1 : (i <= 2*L_common^2 ? theta2 : theta3)
            a = i <= L_common^2 ? a1 : (i <= 2*L_common^2 ? a2 : a3)
            part_with_gamma = i <= L_common^2 ? part_with_gamma_theta_1 : (i <= 2*L_common^2 ? part_with_gamma_theta_2 : part_with_gamma_theta_3)
            psl = i <= L_common^2 ? p_s_l_line[s, f_i] : (i <= 2*L_common^2 ? p_s_l_wave[s, f_i] : p_s_l_cover[s, f_i])
            
            v = 1 / v_var[findfirst(k -> S_div[k,1] <= s <= S_div[k,2], 1:size(S_div,1))]
            pslk = p_s_l_k_vs_combined(s, f_i, q_i, v, theta, a, L_common, w, L1, mu, ae, gamma, type_symbol, part_with_gamma)
            m_c_si_vs = M_C(f_i, v, step, type_symbol, p1_arr, p2_arr)
            
            summ_r1_n += P_s * psl * pslk
            summ_r2_n += m_c_si_vs * P_s * psl * pslk
        end

        summ_r1_ns[idx] = summ_r1_n
        summ_r2_ns[idx] = summ_r2_n
    end

    comp_r1 = 1
    summ_r2 = 0
    for idx = 1:length(s_values)
        comp_r1 *= (1 - summ_r1_ns[idx])
        summ_r2 += (summ_r2_ns[idx] * comp_r1 / (1 - summ_r1_ns[idx]))
    end

	obj_value = summ_r2
    formatted_obj = @sprintf("%.30f", obj_value)
    open("ipopt_sigm_log.txt", "a") do log_file
        println(log_file, "Obj: $formatted_obj, x: $v_var")
    end
    return obj_value
end

function SigmoidRiskFunctionEveryMetr(v_var, S_div, L_common, L1, q_arr, f_arr, step_arr, ae_arr, gamma_arr, hi_arr, theta1, theta2, theta3, a1, a2, a3, p1_arr, p2_arr, p_s_l_line, p_s_l_wave, p_s_l_cover, P_s, part_with_gamma_theta_1, part_with_gamma_theta_2, part_with_gamma_theta_3, w, mu, sum_d_arr)
    s_values = Int[]
    for k = 1:size(S_div, 1)
        append!(s_values, S_div[k,1]:S_div[k,2])
    end

    summ_r1_ns = zeros(eltype(v_var), length(s_values))
    summ_r2_ns = zeros(eltype(v_var), length(s_values))

    @Threads.threads for idx = 1:length(s_values)
        s = s_values[idx]
        summ_r1_n = 0
        summ_r2_n = 0

        for i = 1:3*L_common^2
            q_i = q_arr[i]
            f_i = f_arr[i]
            step = step_arr[i]
            value = s - sum_d_arr[f_i]
            ae, gamma, hi = value < 1 ? (Float64(0), Float64(0), Float64(0)) : (ae_arr[value], gamma_arr[value], hi_arr[value])
            type_symbol = i <= L_common^2 ? :line : (i <= 2*L_common^2 ? :wave : :cover)
            theta = i <= L_common^2 ? theta1 : (i <= 2*L_common^2 ? theta2 : theta3)
            a = i <= L_common^2 ? a1 : (i <= 2*L_common^2 ? a2 : a3)
            part_with_gamma = i <= L_common^2 ? part_with_gamma_theta_1 : (i <= 2*L_common^2 ? part_with_gamma_theta_2 : part_with_gamma_theta_3)
            psl = i <= L_common^2 ? p_s_l_line[s, f_i] : (i <= 2*L_common^2 ? p_s_l_wave[s, f_i] : p_s_l_cover[s, f_i])
            
            v = v_var[idx]
            pslk = p_s_l_k_vs_combined(s, f_i, q_i, v, theta, a, L_common, w, L1, mu, ae, gamma, type_symbol, part_with_gamma)
            m_c_si_vs = M_C(f_i, v, step, type_symbol, p1_arr, p2_arr)
            
            summ_r1_n += P_s * psl * pslk
            summ_r2_n += m_c_si_vs * P_s * psl * pslk
        end

        summ_r1_ns[idx] = summ_r1_n
        summ_r2_ns[idx] = summ_r2_n
    end

    comp_r1 = 1
    summ_r2 = 0
    for idx = 1:length(s_values)
        comp_r1 *= (1 - summ_r1_ns[idx])
        summ_r2 += (summ_r2_ns[idx] * comp_r1 / (1 - summ_r1_ns[idx]))
    end

    return summ_r2
end
