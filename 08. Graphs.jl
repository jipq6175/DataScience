# julia graphs

cd("G:\\My Drive\\18. Github Repo\\DataScience");

# load library
using LightGraphs, Plots, VegaLite, LinearAlgebra, SparseArrays, DataFrames, VegaDatasets, MatrixNetworks

# load data
airports = dataset("airports");
flightsairport = dataset("flights-airport");

# convert to airport
airportsdf = DataFrame(airports);
flightsairportdf = DataFrame(flightsairport);

# show head
first(airportsdf, 10);
first(flightsairportdf, 10);

# only include airport thats in the :origin and :destination
uairports = unique([flightsairportdf[!, :origin]; flightsairportdf[!, :destination]]);
subidx = map(x -> findall(airportsdf[!, :iata] .== uairports[x])[1], 1:length(uairports));
airportsdf_sub = airportsdf[subidx, :];
first(airportsdf_sub, 10)


# build the adjacency matrix
ei_ids = findfirst.(isequal.(flightsairportdf[!,:origin]), [uairports]);
ej_ids = findfirst.(isequal.(flightsairportdf[!,:destination]), [uairports]);
edgeweights = flightsairportdf[!,:count]

A = sparse(ei_ids,ej_ids,1,length(uairports),length(uairports))
A = max.(A,A')

# grapg
G = SimpleGraph(10) #SimpleGraph(nnodes,nedges)
add_edge!(G,7,5) #modifies graph in place.
add_edge!(G,3,5)
add_edge!(G,5,2)

cc = scomponents(A)

degrees = sum(A,dims=2)[:]
p1 = plot(sort(degrees,rev=true),ylabel="log degree",legend=false,yaxis=:log)
p2 = plot(sort(degrees,rev=true),ylabel="degree",legend=false)
plot(p1,p2,size=(600,300))


maxidx = argmax(degrees)
airportsdf_sub[maxidx, :] # ATL


# plot
us10m = dataset("us-10m")
@vlplot(width=500, height=300) +
@vlplot(
    mark={
        :geoshape,
        fill=:lightgray,
        stroke=:white
    },
    data={
        values=us10m,
        format={
            type=:topojson,
            feature=:states
        }
    },
    projection={type=:albersUsa},
) +
@vlplot(
    :circle,
    data=airportsdf_sub,
    projection={type=:albersUsa},
    longitude="longitude:q",
    latitude="latitude:q",
    size={value=10},
    color={value=:steelblue}
)+
@vlplot(
    :rule,
    data=flightsairport,
    transform=[
        {filter={field=:origin,equal=:ATL}},
        {
            lookup=:origin,
            from={
                data=airportsdf_sub,
                key=:iata,
                fields=["latitude", "longitude"]
            },
            as=["origin_latitude", "origin_longitude"]
        },
        {
            lookup=:destination,
            from={
                data=airportsdf_sub,
                key=:iata,
                fields=["latitude", "longitude"]
            },
            as=["dest_latitude", "dest_longitude"]
        }
    ],
    projection={type=:albersUsa},
    longitude="origin_longitude:q",
    latitude="origin_latitude:q",
    longitude2="dest_longitude:q",
    latitude2="dest_latitude:q"
)


# shortest path
ATL_paths = dijkstra(A, maxidx);
ATL_paths[1][maxidx]
maximum(ATL_paths[1])
@show stop1 = argmax(ATL_paths[1])
@show uairports[stop1]

airportsdf_sub[airportsdf_sub[!, :iata] .== "GST", :];

@show stop2 = ATL_paths[2][stop1]
@show uairports[stop2]
airportsdf_sub[airportsdf_sub[!, :iata] .== "JNU", :]

@show stop3 = ATL_paths[2][stop2]
@show uairports[stop3]
airportsdf_sub[airportsdf_sub[!, :iata] .== "SEA", :]


@show stop4 = ATL_paths[2][stop3]
@show uairports[stop4]
airportsdf_sub[airportsdf_sub[!, :iata] .== "ATL", :]


us10m = dataset("us-10m")
airports = dataset("airports")

@vlplot(width=800, height=500) +
@vlplot(
    mark={
        :geoshape,
        fill="#eee",
        stroke=:white
    },
    data={
        values=us10m,
        format={
            type=:topojson,
            feature=:states
        }
    },
    projection={type=:albersUsa},
) +
@vlplot(
    :circle,
    data=airportsdf_sub,
    projection={type=:albersUsa},
    longitude="longitude:q",
    latitude="latitude:q",
    size={value=5},
    color={value=:gray}
) +
@vlplot(
    :line,
    data={
        values=[
            {airport=:ATL,order=1},
            {airport=:SEA,order=2},
            {airport=:JNU,order=3},
            {airport=:GST,order=4}
        ]
    },
    transform=[{
        lookup=:airport,
        from={
            data=airports,
            key=:iata,
            fields=["latitude","longitude"]
        }
    }],
    projection={type=:albersUsa},
    longitude="longitude:q",
    latitude="latitude:q",
    order={field=:order,type=:ordinal}
)


# Minimum Spanning Tree (MST)
@time ti,tj,tv,nverts = mst_prim(A);

df_edges = DataFrame(:ei=>uairports[ti],:ej=>uairports[tj])


@vlplot(width=800, height=500) +
@vlplot(
    mark={
        :geoshape,
        fill="#eee",
        stroke=:white
    },
    data={
        values=us10m,
        format={
            type=:topojson,
            feature=:states
        }
    },
    projection={type=:albersUsa},
) +
@vlplot(
    :circle,
    data=airportsdf_sub,
    projection={type=:albersUsa},
    longitude="longitude:q",
    latitude="latitude:q",
    size={value=20},
    color={value=:gray}
) +
@vlplot(
    :rule,
    data=df_edges, #data=flightsairport,
    transform=[
        {
            lookup=:ei,
            from={
                data=airportsdf_sub,
                key=:iata,
                fields=["latitude", "longitude"]
            },
            as=["originx", "originy"]
        },
        {
            lookup=:ej,
            from={
                data=airportsdf_sub,
                key=:iata,
                fields=["latitude", "longitude"]
            },
            as=["destx", "desty"]
        }
    ],
    projection={type=:albersUsa},
    longitude="originy:q",
    latitude="originx:q",
    longitude2="desty:q",
    latitude2="destx:q"
)



# PageRank
v = MatrixNetworks.pagerank(A,0.85);
insertcols!(airportsdf_sub,7,:pagerank_value => v)


@vlplot(width=500, height=300) +
@vlplot(
    mark={
        :geoshape,
        fill="#eee",
        stroke=:white
    },
    data={
        values=us10m,
        format={
            type=:topojson,
            feature=:states
        }
    },
    projection={type=:albersUsa},
) +
@vlplot(
    :circle,
    data=airportsdf_sub,
    projection={type=:albersUsa},
    longitude="longitude:q",
    latitude="latitude:q",
    size="pagerank_value:q",
    color={value=:steelblue}
)
