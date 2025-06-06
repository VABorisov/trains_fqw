using TickTock
using Base.Threads
using SpecialFunctions
using Distributions

const _norm = Normal()

function precalculation(S, S_div, theta1, theta2, theta3, muliplier_p_line_s_l, muliplier_p_wave_s_l, muliplier_p_cover_s_l, ae_s, gamma_s, v1s, v2s, v3s, L_common, sum_d_arr, hi_s, b1, b2, mu)

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

            #v1
            v1_arr[k1] = v1s[k]

            #v2
            v2_arr[k1] = v2s[k]

            #v3
            v3_arr[k1] = v3s[k]
            
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
        p1_arr[l] = (atan(b1[1] + b1[3] * (L_common - l + 1) + b1[2] * mu)/pi)+0.5
        p2_arr[l] = cdf(_norm,  b2[1] + b2[3] * (L_common - l + 1) + b2[2] * mu)
    end

    return ae_arr, gamma_arr, hi_arr, v1_arr, v2_arr, v3_arr, p_s_l_line, p_s_l_wave, p_s_l_cover, part_with_gamma_theta_1, part_with_gamma_theta_2, part_with_gamma_theta_3, p1_arr, p2_arr
end

function g1(a1, xmax, v, w, L, Lw, mu, ae, gamma)
    ae_not_equal_0_indicator = Int(ae != 0)
    ae_equal_0_indicator = Int(ae == 0)
    gamma_more_0_indicator = Int(gamma > 0)
    gamma_less_0_indicator = Int(gamma < 0)
    exp1 = a1[1] + 
           a1[2] * ae * log(v * 3.6) + 
           a1[3] * (1 - mu) * log(xmax) * log(v * 3.6) * min(0, gamma) + 
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
	common_part = 4.5e6 * (9/200) * v * step
    
    if p_type == :line
        return p1_arr[f] * 1e5 + common_part
    elseif p_type == :wave
        return p2_arr[f] * 1e5 + common_part
    else
        return 1e5 + common_part
    end

end

function Risks(S_div, L_common, L1, q_arr, f_arr, step_arr, v_var, ae_arr, gamma_arr, hi_arr, theta1, theta2, theta3, a1, a2, a3, p1_arr, p2_arr, p_s_l_line, p_s_l_wave, p_s_l_cover, P_s, part_with_gamma_theta_1, part_with_gamma_theta_2, part_with_gamma_theta_3, w, mu, sum_d_arr)
    tick()

    # Собираем все значения s в один массив
    s_values = Int[]
    for k = 1:size(S_div, 1)
        append!(s_values, S_div[k,1]:S_div[k,2])
    end

    # Инициализируем массивы для хранения промежуточных результатов
    summ_r1_ns = zeros(length(s_values))
    summ_r2_ns = zeros(length(s_values))

    println(typeof(summ_r1_ns))

    # Параллельно вычисляем summ_r1_n и summ_r2_n для каждого s
    @Threads.threads for idx = 1:length(s_values)
        s = s_values[idx]
        summ_r1_n = 0
        summ_r2_n = 0

        for i = 1:3*L_common^2
            q_i = q_arr[i]
            f_i = f_arr[i]
            step = step_arr[i]
            value = s - sum_d_arr[f_i]
            ae, gamma, hi = value < 1 ? (0.0, 0.0, 0.0) : (ae_arr[value], gamma_arr[value], hi_arr[value])
            type_symbol = i <= L_common^2 ? :line : (i <= 2*L_common^2 ? :wave : :cover)
            theta = i <= L_common^2 ? theta1 : (i <= 2*L_common^2 ? theta2 : theta3)
            a = i <= L_common^2 ? a1 : (i <= 2*L_common^2 ? a2 : a3)
            part_with_gamma = i <= L_common^2 ? part_with_gamma_theta_1 : (i <= 2*L_common^2 ? part_with_gamma_theta_2 : part_with_gamma_theta_3)
            psl = i <= L_common^2 ? p_s_l_line[s, f_i] : (i <= 2*L_common^2 ? p_s_l_wave[s, f_i] : p_s_l_cover[s, f_i])
            
            v = v_var[findfirst(k -> S_div[k,1] <= s <= S_div[k,2], 1:size(S_div,1))]
            pslk = p_s_l_k_vs_combined(s, f_i, q_i, v, theta, a, L_common, w, L1, mu, ae, gamma, type_symbol, part_with_gamma)
            m_c_si_vs = M_C(f_i, v, step, type_symbol, p1_arr, p2_arr)
            
            summ_r1_n += P_s * psl * pslk
            summ_r2_n += m_c_si_vs * P_s * psl * pslk
        end

        summ_r1_ns[idx] = summ_r1_n
        summ_r2_ns[idx] = summ_r2_n
    end

    # Последовательно вычисляем comp_r1 и summ_r2
    comp_r1 = 1
    summ_r2 = 0
    for idx = 1:length(s_values)
        summ_r2 += summ_r2_ns[idx] * comp_r1
        comp_r1 *= (1 - summ_r1_ns[idx])
    end

    tock()
    return 1 - comp_r1, summ_r2
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
    v1s = [12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5]
    v2s = [16, 10, 12.5, 10, 10, 12.5, 20, 5, 6, 16]
    v3s = [16.34, 11.5, 12.5, 10.5, 12.5, 10, 16.28, 10, 10, 11.692]
    ae_s = [1 / 2000, 1 / 1500, 1 / 1500, 0, 1 / 800, 1 / 1000, 0, 1 / 600, 0, 0]
    gamma_s = [-0.0001, -0.009, -0.003, 0.01, -0.004, 0.008, 0, 0.005, -0.01, 0.005]
    hi_s = 1
    theta1 = 3.83
    theta2 = 0.2933
    theta3 = 0.41
    a1 = [-7.76, 315.69, 286.88, 0.63, -333.03, 4.32, 0.17, -1.55, 0.2, 0.04]
    a2 = [-5.33, 0.83, 0.56, 1.36, 0.2933]
    a3 = [-1.49, 0.99, -0.16, -0.91, 0.43, 0.41]
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
    ae_arr, gamma_arr, hi_arr, v1_arr, v2_arr, v3_arr, p_s_l_line, p_s_l_wave, p_s_l_cover, part_with_gamma_theta_1, part_with_gamma_theta_2, part_with_gamma_theta_3, p1_arr, p2_arr = precalculation(
    S, S_div,theta1,theta2, theta3, muliplier_p_line_s_l, muliplier_p_wave_s_l, muliplier_p_cover_s_l, ae_s, gamma_s, v1s, v2s,v3s, L_common, sum_d_arr, hi_s, b1, b2, mu)
    v_arrs = v3s
    r2 = Risks(S_div, L_common, L1, q_arr, f_arr, step_arr, v_arrs, ae_arr, gamma_arr, hi_arr, theta1, theta2, theta3, a1, a2, a3, p1_arr, p2_arr, p_s_l_line, p_s_l_wave, p_s_l_cover, P_s, part_with_gamma_theta_1, part_with_gamma_theta_2, part_with_gamma_theta_3, w, mu, sum_d_arr)
    println(r2)
end

main()
