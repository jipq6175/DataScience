# clustering julia

# define the accuracy function
function acc(pred::AbstractVector, truth::AbstractVector)
    return sum(pred .== truth) / length(pred);
end

# load the libraries
using GLMNet, RDatasets, MLBase, Plots, DecisionTree, Distances, NearestNeighbors, Random, LinearAlgebra, LIBSVM

# load iris data
iris = dataset("datasets", "iris");
describe(iris)

X = Matrix{Float64}(iris[!, 1:end-1]);
labels = iris[!, end];
labelsmap = labelmap(labels);
Y = labelencode(labelsmap, labels);

# using the first two classes to perform feature reduction
# similar to 04. dim reduction
# PCA
using ScikitLearn, MultivariateStats, StatsBase, Statistics
x = X[1:100, :];
y = Y[1:100];

# standardize the data and use PCA
data = (x .- mean(x, dims=1)) ./ std(x, dims=1);
p = fit(PCA, data', maxoutdim=2);
P = projection(p);
pca_reduction = P' * data';
scatter(pca_reduction[1, 1:50], pca_reduction[2, 1:50], lab="setosa");
scatter!(pca_reduction[1, 51:end], pca_reduction[2, 51:end], lab="versicolor")
@show norm(data' - reconstruct(p, pca_reduction))


# using tSNE
@sk_import manifold : TSNE;
tsnepy = TSNE(n_components=2 ,perplexity=20.0,early_exaggeration=50);
@time tsne_reduction = tsnepy.fit_transform(data);
scatter(tsne_reduction[1:50, 1], tsne_reduction[1:50, 2], lab="setosa");
scatter!(tsne_reduction[51:100, 1], tsne_reduction[51:100, 2], lab="versicolor")


# using UMAP
using UMAP
debye = pairwise(Euclidean(), data, data, dims=1);
@time umap_reduction = umap(debye.^2, 2);
scatter(umap_reduction[1, 1:50], umap_reduction[2, 1:50], lab="setosa");
scatter!(umap_reduction[1, 51:end], umap_reduction[2, 51:end], lab="cersicolor")

corr = cor(data')
@time umap_reduction = umap(10*corr, 2);  # note that corr.^2 does not work
scatter(umap_reduction[1, 1:50], umap_reduction[2, 1:50], lab="setosa");
scatter!(umap_reduction[1, 51:end], umap_reduction[2, 51:end], lab="cersicolor")


# back to clustering
X;
Y;
labeldecode(labelsmap, Y);

# perclass splitting
# test-train split
function perclass_split(y::AbstractVector, at::T) where T<:Real
    uids = unique(y);
    keepids = [];
    for ui in uids
        curids = findall(y .== ui);
        rowids = randsubseq(curids, at);
        push!(keepids, rowids...);
    end
    return keepids;
end

trainids = perclass_split(y, 0.7);
testids = setdiff(collect(1:length(y)), trainids);

function assign_class(pred::T) where T<:Real
    return argmin(abs.(collect(1:3) .- pred));
end


# lasso and elastic net
path = glmnet(X[trainids,:], y[trainids]);
cv = glmnetcv(X[trainids,:], y[trainids]); # choosing the best lammbda (reg const)
lam = path.lambda[argmin(cv.meanloss)];

path = glmnet(X[trainids, :], Y[trainids], lambda = [lam]);
pred_lasso = GLMNet.predict(path, X[testids, :]);
@show acc(assign_class.(pred_lasso[:,1]), y[testids]) # acc = 1.0


# Ridge
path = glmnet(X[trainids,:], y[trainids], alpha=0.0);
cv = glmnetcv(X[trainids,:], y[trainids], alpha=0.0); # choosing the best lammbda (reg const)
lam = path.lambda[argmin(cv.meanloss)];

path = glmnet(X[trainids, :], Y[trainids], lambda = [lam], alpha=0.0);
pred_ridge = GLMNet.predict(path, X[testids, :]);
@show acc(assign_class.(pred_ridge[:,1]), y[testids])


# Elastic Net
#combination of lasso and ridge
path = glmnet(X[trainids,:], y[trainids], alpha=0.5);
cv = glmnetcv(X[trainids,:], y[trainids], alpha=0.5); # choosing the best lammbda (reg const)
lam = path.lambda[argmin(cv.meanloss)];
path = glmnet(X[trainids, :], Y[trainids], lambda = [lam], alpha=0.5);
pred_ridge = GLMNet.predict(path, X[testids, :]);
@show acc(assign_class.(pred_ridge[:,1]), y[testids])





# Tree Based models
# decision tree
dtclassifier = DecisionTreeClassifier(max_depth=2);
@time DecisionTree.fit!(dtclassifier, X[trainids, :], Y[trainids]);
pred_dt = DecisionTree.predict(dtclassifier, X[testids, :]);
@show acc(pred_dt, Y[testids])

# random forest
rfclassifier = RandomForestClassifier(n_trees=20);
@time DecisionTree.fit!(rfclassifier, X[trainids, :], Y[trainids]);
pred_rf = DecisionTree.predict(rfclassifier, X[testids, :]);
@show acc(pred_rf, Y[testids]);

# xgboost
using XGBoost
@time xgbclassifier = xgboost(X[trainids, :], 100, eta=0.2, max_depth=2, silent=false, label=Y[trainids]);
pred_xgb = XGBoost.predict(xgbclassifier, X[testids, :]);
@show acc(assign_class.(pred_xgb), Y[testids])



# SVM
svm = svmtrain(X[trainids, :]', Y[trainids]);
pred_svm, __ = svmpredict(svm, X[testids, :]');
@show acc(pred_svm, Y[testids])
