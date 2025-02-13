using TickTock
using Base.Threads
using SpecialFunctions
using Distributions
using Ipopt
using JuMP


# Создаём модель с Ipopt
model = Model(Ipopt.Optimizer)

# Устанавливаем решатель линейных систем Pardiso или MA97
set_optimizer_attribute(model, "linear_solver","mumps")  # или "ma97"

# Определяем переменные
@variable(model, x)
@variable(model, y)

# Задаём нелинейную целевую функцию
@NLobjective(model, Min, (x - 2)^2 + (y - 3)^2 + x * y)

# Добавляем нелинейные ограничения
@constraint(model, x + y ≥ 2)
@NLconstraint(model, x^2 + y^2 ≤ 4)

# Решаем задачу
optimize!(model)

# Выводим результаты
println("Статус: ", termination_status(model))
println("x = ", value(x))
println("y = ", value(y))
println("Оптимальное значение целевой функции: ", objective_value(model))
