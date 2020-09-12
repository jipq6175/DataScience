# clustering julia

# load libs
using VegaLite, VegaDatasets, DataFrames, Statistics, JSON, CSV, Distances, Clustering

# loading the data
download("https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv","newhouses.csv");
@time houses = CSV.read("newhouses.csv", DataFrame);

#
cd("G:\\My Drive\\18. Github Repo\\DataScience")
describe(houses);
names(houses)

# vegalite
cali_shape = JSON.parsefile("data/california-counties.json")
VV = VegaDatasets.VegaJSONDataset(cali_shape,"data/california-counties.json")

# plotting using the median house price
@vlplot(width=500, height=300) +
@vlplot(
    mark={:geoshape, fill=:black, stroke=:white},
    data={values=VV, format={type=:topojson, feature=:cb_2015_california_county_20m}},
    projection={type=:albersUsa}) +
@vlplot(:circle, data=houses, projection={type=:albersUsa}, longitude="longitude:q", latitude="latitude:q", size={value=12}, color="median_house_value:q")

#
bucketprice = Int.(div.(houses[!,:median_house_value],50000));
insertcols!(houses,3, :cprice => bucketprice);

@vlplot(width=500, height=300) +
@vlplot(
    mark={:geoshape, fill=:black, stroke=:white},
    data={values=VV, format={type=:topojson, feature=:cb_2015_california_county_20m}},
    projection={type=:albersUsa}) +
@vlplot(:circle, data=houses, projection={type=:albersUsa}, longitude="longitude:q", latitude="latitude:q", size={value=12}, color="cprice:n")


# clustering
X = Matrix{Float64}(houses[!, [:latitude, :longitude]]);
@time C = kmeans(X', 10);
insertcols!(houses, 3, :cluster10 => C.assignments);

@vlplot(width=500, height=300) +
@vlplot(
    mark={:geoshape, fill=:black, stroke=:white},
    data={values=VV, format={type=:topojson, feature=:cb_2015_california_county_20m}},
    projection={type=:albersUsa}) +
@vlplot(:circle, data=houses, projection={type=:albersUsa}, longitude="longitude:q", latitude="latitude:q", size={value=12}, color="cluster10:n")


# using the distances
D = pairwise(Euclidean(), X', X', dims=2);
@time K = kmedoids(D, 10);
insertcols!(houses, 3, :medoid10 => K.assignments);

@vlplot(width=500, height=300) +
@vlplot(
    mark={:geoshape, fill=:black, stroke=:white},
    data={values=VV, format={type=:topojson, feature=:cb_2015_california_county_20m}},
    projection={type=:albersUsa}) +
@vlplot(:circle, data=houses, projection={type=:albersUsa}, longitude="longitude:q", latitude="latitude:q", size={value=12}, color="medoid10:n")


# hierachiecal clustering
@time K = hclust(D);
@time L = cutree(K, k=10);
insertcols!(houses, 3, :hclust => L);

@vlplot(width=500, height=300) +
@vlplot(
    mark={:geoshape, fill=:black, stroke=:white},
    data={values=VV, format={type=:topojson, feature=:cb_2015_california_county_20m}},
    projection={type=:albersUsa}) +
@vlplot(:circle, data=houses, projection={type=:albersUsa}, longitude="longitude:q", latitude="latitude:q", size={value=12}, color="hclust:n")
