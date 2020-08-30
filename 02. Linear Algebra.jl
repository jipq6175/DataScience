
# linear algebra
cd("G:\\My Drive\\18. Github Repo\\DataScience");
# load the packages
using LinearAlgebra, Images, SparseArrays

# Basic operations
A = rand(10, 10);
A = A * A';
b = rand(10);
x = A \ b; # solving linear system

@show norm(b - A * x)

# LU factor
luA = lu(A)

# QR factor
qrA = qr(A);

# cholesky
isposdef(A)
cholA = cholesky(A)


# diag
a = Diagonal([1,2,3])


# images
X1 = load(".\\data\\khiam-small.jpg");
X1
Xgray = Gray.(X1)
Xgrayvalues = Float64.(Xgray)

# svd
sSVD_V = svd(Xgrayvalues);

# using the top 10 singular values
global img = zeros(size(Xgrayvalues));
for i = 1:50
    global img += sSVD_V.S[i] * sSVD_V.U[:, i] * sSVD_V.V[:, i]';
end

Gray.(img)

@show norm(Xgrayvalues - img)

#
