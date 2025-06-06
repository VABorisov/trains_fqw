using TickTock
using Base.Threads
using SpecialFunctions
using Distributions
using Printf

function precalculation(S, S_div, theta1, theta2, theta3, muliplier_p_line_s_l, muliplier_p_wave_s_l, muliplier_p_cover_s_l, ae_s, gamma_s, L_common, sum_d_arr, hi_s)

    ae_arr = zeros(S, 1)
    gamma_arr = zeros(S, 1)
    hi_arr = zeros(S, 1)
    zeta_mat = zeros(S, L_common)
    v1_arr = zeros(S, 1)
    v2_arr = zeros(S, 1)
    v3_arr = zeros(S, 1)
    part_with_gamma_theta_1 = zeros(L_common, 1)
    part_with_gamma_theta_2 = zeros(L_common, 1)
    part_with_gamma_theta_3 = zeros(L_common, 1)
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
    end

    return ae_arr, gamma_arr, hi_arr, p_s_l_line, p_s_l_wave, p_s_l_cover, part_with_gamma_theta_1, part_with_gamma_theta_2, part_with_gamma_theta_3
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
    gamma_more_0_indicator = Int(gamma > 0);
    exp_val = a2[1] + a2[2] * mu^2 + 
              a2[3] * log(v * 3.6) * gamma_more_0_indicator + 
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

function p1(b1, x, xmax, v, w, L, Lw, mu,  ae, gamma)
    res = (atan(b1[1] + b1[2] * x + b1[3] * mu)/pi)+0.5
    return res
end

function p2(b2, x, xmax, v, w, L, Lw, mu, ae, gamma)
    x = b2[1] + b2[2] * x + b2[3] * mu
    res = cdf(Normal(), x)
    return res
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

function M_C(hi, b, q, f, v, L, w, Lw, mu, ae, gamma, step, p_type)
    res = if p_type == :line
        p1(b, q,L-f+1, v, w, L, Lw, mu, ae, gamma) * 10 ^ 5 + 4.5E6 * (9/200) * v * step
    elseif p_type == :wave
        p2(b, q,L-f+1, v, w, L, Lw, mu, ae, gamma) * 10 ^ 5 + 4.5E6 * (9/200) * v * step
    elseif p_type == :cover
        10 ^ 5 + 4.5E6 * (9/200) * v * step
    end
    return res 

end

function Risks(S_div, L_common, L1, q_arr, f_arr, step_arr, v_arrs, ae_arr, gamma_arr, hi_arr, theta1, theta2, theta3, a1, a2, a3, b1, b2, p_s_l_line, p_s_l_wave, p_s_l_cover, P_s, part_with_gamma_theta_1, part_with_gamma_theta_2, part_with_gamma_theta_3, w, mu, sum_d_arr)

    comp_r1 = ones(length(v_arrs)) 
    lk = ReentrantLock()
    
    for k = 1:size(S_div , 1)
        @Threads.threads  for s = S_div[k,1]:S_div[k,2]
	        summ_r1_n = zeros(length(v_arrs))
	        @inbounds for i = 1:3 * L_common^2
	            q_i = q_arr[i]
	            f_i = f_arr[i]
	            step = step_arr[i]
	            value = s - sum_d_arr[f_i]
	            ae, gamma, hi = value < 1 ? (0, 0, 0) : (ae_arr[value], gamma_arr[value], hi_arr[value])
	            type_symbol = i <= L_common^2 ? :line : (i <= 2 * L_common^2 ? :wave : :cover)
	            theta = i <= L_common^2 ? theta1 : (i <= 2 * L_common^2 ? theta2 : theta3)
	            a = i <= L_common^2 ? a1 : (i <= 2 * L_common^2 ? a2 : a3)
	            b = i <= L_common^2 ? b1 : (i <= 2 * L_common^2 ? b2 : 0)
	            part_with_gamma = i <= L_common^2 ? part_with_gamma_theta_1 : (i <= 2 * L_common^2 ? part_with_gamma_theta_2 : part_with_gamma_theta_3)
	            psl = i <= L_common^2 ? p_s_l_line[s, f_i] : (i <= 2 * L_common^2 ? p_s_l_wave[s, f_i] : p_s_l_cover[s, f_i])
	            @inbounds for idx in 1:length(v_arrs)
	                v = v_arrs[idx][k]
	                pslk = p_s_l_k_vs_combined(s, f_i, q_i, v, theta, a, L_common, w, L1, mu, ae, gamma, type_symbol, part_with_gamma)
	                summ_r1_n[idx] += P_s * psl * pslk
	            end
	        end
	        lock(lk)
	        @inbounds for idx in 1:length(comp_r1)
	            comp_r1[idx] *= (1 - summ_r1_n[idx])
	        end
	        unlock(lk)
	    end
    end
    return 1 .- comp_r1
end

function main() 
    Q1_strict = 1.8E11
    Q2_strict = 3E9
    p_1 = 150
    p_2 = 38
    p_3 = 46
    p_common = 246
    L0 = 2
    L1 = 58
    S = 250000
    w = 5000
    S_div = [
        1       50000;
        50001   60000;
        60001   70000;
        70001   100000;
        100001  110000;
        110001  150000;
        150001  200000;
        200001  205000;
        205001  220000;
        220001  250000
    ]
    v5s = fill(5, 10)
    v6s = fill(6, 10)
    v7s = fill(7, 10)
    v8s = fill(8, 10)
    v9s = fill(9, 10)
    v10s = fill(10, 10)
    v11s = fill(11, 10)
    v12s = fill(12, 10)
    v13s = fill(13, 10)
    v14s = fill(14, 10)
    v15s = fill(15, 10)
    v16s = fill(16, 10)
    v17s = fill(17, 10)
    v18s = fill(18, 10)
    v19s = fill(19, 10)
    v20s = fill(20, 10)
    ae_s = [1 / 1000, 1 / 1000, 1 / 1000, 1 / 1000, 1 / 1000, 1 / 1000, 1 / 1000, 1 / 1000, 1 / 1000, 1 / 1000]
    gamma_s = [0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008]
    hi_s = 1
    theta1 = 3.83
    theta2 = 0.2933
    theta3 = 0.41
    a1 = [-7.76, 315.69, 286.88, 0.63, -333.03, 4.32, 0.17, -1.55, 0.2, 0.04]
    a2 = [-5.33, 0.83, 0.56, 1.36]
    a3 = [-1.49, 0.99, -0.16, -0.91, 0.43]
    b1 = [-7.16, 1.98, 2.05]
    b2 = [-2.43, 1.87, 0.19]
    L_common = L0 + L1
    d = vcat([20, 20], fill(14, L_common - 2))
    mu = (w / (69 * L1)) - (1 / 3)
    Q1 = (p_1 + p_2 + p_3) / (1000 * Q1_strict)
    Q2 = (p_common - p_1 - p_2 - p_3) / (1000 * Q2_strict)
    sum_d_arr = [sum(d[1:(l-1)]) for l in 1:L_common]
    P_s = 1 - exp(-(Q1 * L_common + Q2))
    muliplier_p_line_s_l = (1 / L_common) * (p_1 / (p_1 + p_2))
    muliplier_p_wave_s_l = (1 / L_common) * (p_2 / (p_1 + p_2))
    muliplier_p_cover_s_l = 1 / L_common
    q_arr = q_func(L_common)
    f_arr = f_func(L_common)
    step_arr = step_func(L_common, q_arr, f_arr)
    ae_arr, gamma_arr, hi_arr, p_s_l_line, p_s_l_wave, p_s_l_cover, part_with_gamma_theta_1, part_with_gamma_theta_2, part_with_gamma_theta_3 = precalculation(
    S, S_div,theta1,theta2, theta3, muliplier_p_line_s_l, muliplier_p_wave_s_l, muliplier_p_cover_s_l, ae_s, gamma_s, L_common, sum_d_arr, hi_s)
    v_arrs = [v5s, v6s, v7s, v8s, v9s, v10s, v11s, v12s, v13s, v14s, v15s, v16s, v17s, v18s, v19s, v20s]
    r1 = Risks(S_div, L_common, L1, q_arr, f_arr, step_arr, v_arrs, ae_arr, gamma_arr, hi_arr, theta1, theta2, theta3, a1, a2, a3, b1, b2, p_s_l_line, p_s_l_wave, p_s_l_cover, P_s, part_with_gamma_theta_1, part_with_gamma_theta_2, part_with_gamma_theta_3, w, mu, sum_d_arr)
    for v in r1
        @printf("%.30f\n", v)
    end
end

main()
