
# some simple tests with periodic NN style functoins for central pattern generators.
# The goal is to effectively learn the parameters of sine and cosine functions via gradient descent on a time series
using Plots

arr = zeros(Float32,1000)
arr2 = zeros(Float32, 1000)
arr3 = zeros(Float32, 1000)
for i in 1:1000
    arr[i] = sin(0.05 * i)
    arr2[i] = cos(0.05 * i)
    arr3[i] = tan(0.05 * i)
end
plot(arr)
plot!(arr2)
plot!(arr3)

using Flux.Tracker: update!
data = [sin(0.05f0* i) for i in 1:1000]

prediction(period,i) = sin(i * period)

MSE(x,pred) = (x .- pred).^2 # i.e. flux mse loss
loss(data, preds) = mean(MSE.(data, preds))
period = param(0.001)
periods = []
ls = []
predslist = []
grads = []
for i in 1:100
    global period
    println("epoch $i, period: $period")
    preds = [prediction(period,j) for j in 1:1000]
    l = loss(data, preds)
    gs = Tracker.gradient(()->loss(data,preds), params(period))
    #period -= 0.01 * grad
    #period = period.data
    #period = param(period)
    grad = min(max(gs[period].data,1f0),-1f0)
    println("grad: $grad")
    update!(period, -0.01 *grad)
    push!(periods, period.data)
    push!(ls, l.data)
    push!(predslist, Tracker.data.(preds))
    push!(grads, grad)

end
plot(ls)
plot(periods)
plot(predslist[10])
plot(data)
plot(grads)

W = rand(2, 5)
b = rand(2)

prediction(x) = W*x .+ b

function loss(x, y)
  ŷ = prediction(x)
  sum((y .- ŷ).^2)
end

x, y = rand(5), rand(2) # Dummy data
loss(x, y) # ~ 3

# for rhythmical NNs
period = param(0.001)
periods = []
ls = []
predslists = []
grads = []
datalists =[]

for i in 1:100

    global period,L
    println("epoch $i, period: $period")
    for j in 1:40
        predslist = []
        datalist = []
        pred = prediction(period, j)
        data = sin(0.1 * j)
        L = MSE(pred, data)
        gs = Tracker.gradient(()->MSE(pred, data), params(period))
        grad = min(max(gs[period].data,1f0),-1f0)
        #grad = gs[period]
        print("grad: $grad")
        #println("grad: $grad")
        update!(period, -0.001 *grad)
        push!(predslist, pred.data)
        #push!(grads, grad)
        push!(datalist, data)
        println(period)
        println("L: $L")
    end
    push!(datalists, datalist)
    push!(predslists, predslist)
    datalist = []
    predslist = []
    println("outside: $period")
    println("outside l: $L")
    push!(periods, period.data)
    push!(ls, L.data)

end
plot(ls)
plot(periods)

d = [sin(0.001 * i) for i in 1:10000]
plot(d)
period = param(0.01)
pred2(period) = [sin(period*i) for i in 1:10000]
loss(x,y) = mean((x .- y).^2)
periods = []
ls = []
grads = []
preds = []
##
for i in 1:1000
    global period
    l = loss(d, pred2(period))
    pred = pred2(period)
    push!(preds, Tracker.data.(pred))
    gs = Tracker.gradient(()-> loss(d, pred2(period)), params(period))
    grad = gs[period]
    update!(period, -0.001 * grad)
    push!(ls, l.data)
    push!(periods, period.data)
    push!(grads, grad.data)
end
plot(ls)
plot(periods)
plot(grads)
plot(d)
plot!(preds[end])
mutable struct CPG
    i::Int
    p::AbstractFloat
end

function forward(c::CPG)
    pred =  sin(c.i * c.p)
    c.i = c.i+1
    return pred
end

a = CPG(0,0.01)
forward(a)
a.i
##
using Flux
params(a)
d = [sin(0.001 * i) for i in 1:10000]
plot(d)
period = param(0.01)
pred2(c)
loss(x,y) = mean((x .- y).^2)
periods = []
ls = []
grads = []
preds = []
##
for i in 1:1000
    global period
    l = loss(d, pred2(period))
    pred = pred2(period)
    push!(preds, Tracker.data.(pred))
    gs = Tracker.gradient(()-> loss(d, pred2(period)), params(period))
    grad = gs[period]
    update!(period, -0.001 * grad)
    push!(ls, l.data)
    push!(periods, period.data)
    push!(grads, grad.data)
end
plot(ls)
plot(periods)
plot(grads)
plot(d)
plot!(preds[end])