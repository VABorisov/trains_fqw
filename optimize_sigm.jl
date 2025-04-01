using Base.Threads
using Distributions
using Ipopt
using JuMP
using SpecialFunctions
using TickTock
using ForwardDiff
include("upd_risk_func.jl")

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
    T = 20000
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
    S_lengths = S_div[:,2] .- S_div[:,1]  
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
    ipopt = optimizer_with_attributes(Ipopt.Optimizer,
    "print_level" => 5,  
    "linear_solver" => "mumps",
    "bound_relax_factor" => 0.0
    )
    model = Model(ipopt)
    @variable(model, (1/V_const) .<= x[1:size(S_div , 1)] .<= (1/v_const))
    for i in 1:size(S_div, 1)
        set_start_value(x[i], 0.1)
    end
    
    function create_objective_closure(
      S_div, L_common, L1, q_arr, f_arr, step_arr, ae_arr, gamma_arr, hi_arr, theta1, theta2, theta3, a1, a2, a3, p1_arr, p2_arr, p_s_l_line, p_s_l_wave, p_s_l_cover, P_s, part_with_gamma_theta_1, part_with_gamma_theta_2, part_with_gamma_theta_3, w, mu, sum_d_arr
    )
        function objective_closure(x::T...) where {T <: Real}
            x_vector = collect(x)
            result = SigmoidRiskFunction(
                x_vector, S_div, L_common, L1, q_arr, f_arr, step_arr, ae_arr, gamma_arr, hi_arr, theta1, theta2, theta3, a1, a2, a3, p1_arr, p2_arr, p_s_l_line, p_s_l_wave, p_s_l_cover, P_s, part_with_gamma_theta_1, part_with_gamma_theta_2, part_with_gamma_theta_3, w, mu, sum_d_arr
            )
            
            return result
        end
        
        return objective_closure
    end

    objective_closure = create_objective_closure(
    S_div, L_common, L1, q_arr, f_arr, step_arr, ae_arr, gamma_arr, hi_arr, theta1, theta2, theta3, a1, a2, a3, p1_arr, p2_arr, p_s_l_line, p_s_l_wave, p_s_l_cover, P_s, part_with_gamma_theta_1, part_with_gamma_theta_2, part_with_gamma_theta_3, w, mu, sum_d_arr
    )
 
    JuMP.register(
    model,
    :my_objective,
    size(S_div, 1),
    objective_closure;
    autodiff = true
    )

    @NLobjective(
        model,
        Min,
        my_objective(x...)
    )

    @constraint(model, sum(S_lengths[i] * x[i] for i in 1:size(S_lengths, 1)) <= T)
    optimize!(model)
    println(termination_status(model))
    println(objective_value(model))
    v = 1 ./ value.(x)
    println(v)

    v_arr = zeros(Float64, S)
    a = 0.1
       
    
    for k = 1:size(S_div, 1)
        for k1 = S_div[k,1]:S_div[k,2]
            v_arr[k1] = v[k]
        end
    end

    for i = 1:S - 1
		if v_arr[i + 1] > v_arr[i]
			j = i
            v_target = v_arr[i]
            while j >= 1
                new_v = sqrt(v_arr[j + 1]^2 - 2 * a)
                if new_v < v_target
                    break
                end
                v_arr[j] = new_v
                j -= 1
            end
        elseif v_arr[i + 1] < v_arr[i]
            j = i + 1
            v_target = v_arr[i + 1]
            while j <= S
                new_v = sqrt(v_arr[j - 1]^2 - 2 * a)
                if new_v < v_target
                    break
                end
                v_arr[j] = new_v
                j += 1
            end
        end
    end
    open("speeds.txt", "w") do file
		for v in v_arr
			write(file, string(v), "\n")
		end
    end

    res_recalc = SigmoidRiskFunctionEveryMetr(
                    v_arr, S_div, L_common, L1, q_arr, f_arr, step_arr, ae_arr, gamma_arr, hi_arr, theta1, theta2, theta3, a1, a2, a3, p1_arr, p2_arr, p_s_l_line, p_s_l_wave, p_s_l_cover, P_s, part_with_gamma_theta_1, part_with_gamma_theta_2, part_with_gamma_theta_3, w, mu, sum_d_arr
                )
    println(res_recalc) 
end

main()
