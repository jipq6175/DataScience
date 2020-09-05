# statistics
cd("G:\\My Drive\\18. Github Repo\\DataScience");
# load the libraries
using Statistics, StatsBase, Plots, StatsPlots, Distributions, LinearAlgebra, MLBase, HypothesisTests, PyCall, DataFrames, RDatasets

# Load data and clean
D = dataset("datasets", "faithful");
@show names(D);
describe(D)

# plot
eruptions = D[!, :Eruptions];
scatter(eruptions, lab="Eruptions");
waittime = D[!, :Waiting];
scatter!(waittime, lab="waittime")

# boxplot
boxplot(["eruption length"], eruptions, legend=false, size=(200,400), whisker_width=1, ylabel="time in minutes")

# histogram
histogram(eruptions, lab="eruptions", bins=:sqrt)


# fit using kernel density
using KernelDensity
p = kde(eruptions);

# fits
histogram(eruptions,label="eruptions", bins=:sqrt)
plot!(p.x,p.density .* length(eruptions)*0.4, linewidth=3,color=2,label="kde fit")
# nb of elements*bin width

histogram(eruptions,bins=:sqrt,label="eruptions")
plot!(p.x,p.density .* length(eruptions) .*0.2, linewidth=3,color=2,label="kde fit")


#
myrandomvector = randn(100_000)
histogram(myrandomvector)
p=kde(myrandomvector)
plot!(p.x,p.density .* length(myrandomvector) .*0.1, linewidth=3,color=2,label="kde fit") # nb of elements*bin width


# Probability distributions
d = Normal()
myrandomvector = rand(d,100000)
histogram(myrandomvector)
p=kde(myrandomvector)
plot!(p.x,p.density .* length(myrandomvector) .*0.1, linewidth=3,color=2,label="kde fit") # nb of elements*bin width

b = Binomial(40)
myrandomvector = rand(b,1000000)
histogram(myrandomvector)
p=kde(myrandomvector)
plot!(p.x,p.density .* length(myrandomvector) .*0.5,color=2,label="kde fit") # nb of elements*bin width


# fit distributions
x = rand(1000)
d = fit(Normal, x)
myrandomvector = rand(d,1000)
histogram(myrandomvector,nbins=20,fillalpha=0.3,label="fit")
histogram!(x,nbins=20,linecolor = :red,fillalpha=0.3,label="myvector")


x = eruptions
d = fit(Normal, x)
myrandomvector = rand(d,1000)
histogram(myrandomvector,nbins=20,fillalpha=0.3)
histogram!(x,nbins=20,linecolor = :red,fillalpha=0.3)


# T-Test
ttest = OneSampleTTest(myrandomvector)


# MLBase
gt = [1,1,1,1,1,1,1,2];
pred = [1,1,2,2,1,1,1,1];
C = confusmat(2, gt, pred);
C = C ./ sum(C, dims=2)
correctrate(gt, pred)

#
scatter(eruptions, waittime)
