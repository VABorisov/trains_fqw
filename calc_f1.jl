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

function g1(a1, xmax, w, L, Lw, mu, ae, gamma)
    ae_not_equal_0_indicator = Int(ae != 0)
    ae_equal_0_indicator = Int(ae == 0)
    gamma_more_0_indicator = Int(gamma > 0)
    gamma_less_0_indicator = Int(gamma < 0)
    exp1 = a1[2] * ae  + 
           a1[3] * (1 - mu) * log(xmax) *  min(0, gamma) + 
           a1[5] * (1 - mu)^2 * log(xmax) *  min(0, gamma) + 
           a1[7] * gamma_more_0_indicator * mu * log(xmax)
    exp2 = a1[9] * gamma_less_0_indicator * log(xmax) + 
           a1[10] * gamma_more_0_indicator * (log(xmax))^2
    result = ae_not_equal_0_indicator * exp1 + 
                 ae_equal_0_indicator * exp2
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

function CalcF1(S, S_div, L_common, L1, q_arr, f_arr, step_arr, ae_arr, gamma_arr, hi_arr, theta1, theta2, theta3, a1, a2, a3, p1_arr, p2_arr, p_s_l_line, p_s_l_wave, p_s_l_cover, P_s, part_with_gamma_theta_1, part_with_gamma_theta_2, part_with_gamma_theta_3, w, mu, sum_d_arr)

	results = zeros(S,1)
    for s = 1:S
   		f_i = 60

   		value = s - sum_d_arr[f_i]
   		ae, gamma, hi = value < 1 ? (Float64(0), Float64(0), Float64(0)) : (ae_arr[value], gamma_arr[value], hi_arr[value])

		
   		results[s] = g1(a1, L_common-f_i+1, w, L_common, L1, mu, ae, gamma)
    end

    open("f1_60.csv", "w") do file

       write(file, "meter,f1\n")
   
       for i in 1:S
           write(file, "$(i),$(results[i])\n")
       end
    end
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
    v_const = 5
    V_const = 25
    T = 15000
    ae_s = [1 / 2000, 1 / 1500, 1 / 1500, 0, 1 / 800, 1 / 1000, 0, 1 / 600, 0, 0]
    gamma_s = [-0.0001, -0.009, -0.003, 0.01, -0.004, 0.008, 0, 0.005, -0.01, 0.005]
    hi_s = 1
    theta1 = 3.83
    theta2 = 0.3
    theta3 = 0.41
    a1 = [-7.76, 315.69, 286.88, 0.63, -333.03, 4.32, 0.17, -1.55, 0.2, 0.04]
    a2 = [-6.4, 1.01, 0.68, 1.48]
    a3 = [-1.49, 0.99, -0.16, -0.91, 0.43, 0.41]
    b1 = [-7.16, 2.05, 1.98]
    b2 = [-2.43, 0.18, 1.87]
    L_common = L0 + L1
    S_lengths = S_div[:,2] .- S_div[:,1] .+ 1 
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
    ae_arr,
    gamma_arr, 
    hi_arr, 
    p_s_l_line, 
    p_s_l_wave, 
    p_s_l_cover,
    part_with_gamma_theta_1, 
    part_with_gamma_theta_2, 
    part_with_gamma_theta_3,
    p1_arr,
    p2_arr = precalculation(S,
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
   CalcF1(S, S_div, L_common, L1, q_arr, f_arr, step_arr, ae_arr, gamma_arr, hi_arr, theta1, theta2, theta3, a1, a2, a3, p1_arr, p2_arr, p_s_l_line, p_s_l_wave, p_s_l_cover, P_s, part_with_gamma_theta_1, part_with_gamma_theta_2, part_with_gamma_theta_3, w, mu, sum_d_arr)
                                
end

main()
