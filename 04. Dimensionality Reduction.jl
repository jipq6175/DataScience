# 04 dimension reduction

# using the libraries
using UMAP, XLSX, VegaDatasets, DataFrames, MultivariateStats, RDatasets, StatsBase, Statistics, LinearAlgebra, Plots, ScikitLearn, MLBase, Distances

# import the data
C = DataFrame(VegaDatasets.dataset("cars"));
describe(C);

# clean the data
dropmissing!(C);
M = Matrix{Float64}(C[!, 2:7]);
n = names(C)[2:end];

# For MLBase and label map
# USA -> 1
# Europe -> 2 etc...
car_origin = C[!, :Origin];
unique(car_origin);
carmap = labelmap(car_origin);
uniqueids = labelencode(carmap, car_origin)

# PCA approximate the matrix
# Center and standardize the data
data = M;
data = (data .- mean(data, dims=1)) ./ std(data, dims=1);

# fit the data
p = fit(PCA, data', maxoutdim=2)

# projection
P = projection(p);
P' * (data[1, :] - mean(p))

#
Yte = P' * data'
Yte = MultivariateStats.transform(p, data')

# reconstruct the matrix
Xr = reconstruct(p, Yte);
data' - Xr;
@show norm(data' - Xr)
# the approximation gets better when more maxoutdims are used.


# plotting
scatter(Yte[1,:],Yte[2,:])

scatter(Yte[1,car_origin.=="USA"],Yte[2,car_origin.=="USA"],color=1,label="USA")
scatter!(Yte[1,car_origin.=="Japan"],Yte[2,car_origin.=="Japan"],color=2,label="Japan")
scatter!(Yte[1,car_origin.=="Europe"],Yte[2,car_origin.=="Europe"],color=3,label="Europe")
xlabel!("pca component1")
ylabel!("pca component2")

# three dim maxxout
p = fit(PCA,data',maxoutdim=3)
Yte = MultivariateStats.transform(p, data')
scatter3d(Yte[1,:],Yte[2,:],Yte[3,:],color=uniqueids,legend=false)


# tsne
# @sk_import manifold : TSNE
tfn = TSNE(n_components=3 ,perplexity=20.0,early_exaggeration=50)
Y2 = tfn.fit_transform(data);
scatter(Y2[:,1],Y2[:,2],Y2[:,3],color=uniqueids,legend=false,size=(400,300),markersize=3)


# umap
L = cor(data,data,dims=2)
@time embedding = umap(L, )

scatter(embedding[1,:],embedding[2,:],color=uniqueids)


# different distance
@time L2 = pairwise(Euclidean(), data, data,dims=1)
embedding = umap(-L, 2)
scatter(embedding[1,:],embedding[2,:],color=uniqueids)
