using DelimitedFiles, Random, Distributions, StatsBase, LightGraphs

function Parameters(aid::Int64)
    ls_td = [(0.00005,0.0005),(0.0005,0.005),(0.005,0.05),(0.05,0.5),(0.1,1),(0.15,1.5),(0.2,2),(0.25,2.5),(0.3,3),(0.35,3.5),(0.4,4)];
    tplenv_ls = [("Complete","Lowest"),("Linear","Lowest"),("Linear","Mid"),("Linear","Highest"),("Grid","Lowest"),("Grid","Mid"),("Grid","Highest"),("SmallWorld","Lowest"),("SmallWorld","Mid"),("SmallWorld","Highest"),("Tree","Lowest"),("Tree","Mid"),("Tree","Highest")];

    A = collect(Iterators.product(tplenv_ls,ls_td));

    tplenv,lstd=A[aid];
    td,w_adjust = lstd;
    topology,autocorr = tplenv;
    return topology,autocorr,td,w_adjust
end;

function Initial(n_sp::Int,n_com::Int,pinit::Float64)
    #=
    pinit: the initial Pik for all i and k. pinit*n_sp has to be <= 1.
    The example they used in the paper was n_sp = 20, n_com = 20
    =#
    Q = zeros(Float64, n_sp, n_com) .+ Float64(pinit);
    return Q
end;

function ReadMat(filename,w_adjust)
    # the imported matrix file is for total dispersal 0.1. w_adjust is a parameter to adjust the total dispersal rate by multiplying the matrix.
    mat = readdlm(filename, ' ', Float64) * w_adjust;
    return mat
end;

function ReadEnvs(filename)
    E = readdlm(filename);
    E = E';
    return E
end;

function ReproductionRate(n_sp,n_com,h,E_ls)
    X = zeros(Float64, n_sp, n_com);
    for k in (1:n_com)
        X[:,k] = 0:1/(n_sp-1):1;
    end
    E = repeat(E_ls,n_sp,1); # assign the rest of elements so that all elements within each column has are the same.
    D = E .- X;
    C = broadcast(exp, (-(D.*D) ./ (2*h))) ./ sqrt(2 * h * pi);
    #C = C ./ sum(C,dims=2); # normalizing the fitness distribution such that each species has equal regional fitness.
    return C
end;

function WrightFisher(Q)
    # all individuals die and will be replaced by offspring
    Q = zeros(size(Q));
    return Q
end;

function Update(Q,A,C,nu,Preg,n_com)
    #=
    <Variables>
    Matrix Q (Qik): the fraction of sites occupied by species i in community k.
    <Parameters>
    Matrix A (alk): dispersal rate from community l to k. diag(A) = 0.
    Matrix C (cik): per capita potential reproductive rate of species i in community k.
    =#
    S = transpose(sum(A,dims=2)); # row sum of the matrix A, transposed into a row vector
    I = ones(size(Q));
    Z = I - repeat(S,n_sp,1); # repeat() is to duplicate the row vector S and make a matrix
    if any(Z .< 0)
        Z[findall(x->abs(x)<10^(-10),Z)] .= 0;
        #if any(Z .< 0)
        #    println("large negative value exists in Z");
        #    return "NaN"
        #end
    end
    q = (C.*Q) * A + (Z.*(C.*Q));

    # Death Process
    Q = WrightFisher(Q);

    # Stochastic sampling of new settlers
    q = q ./ sum(q,dims=1);
    # immigration process
    Vk = 1;
    if nu > 0
        Q .+= nu/n_sp; # Adding immigrants. sum_i(nu/n_sp) = nu. nu is fraction of immigrants per local community.
        Vk -= nu; # fraction of empty sites per community
    end
    Q .+= Vk*q;
    return Q
end;

function calcDiff(Q,prevQ)
    diff = sum(broadcast(abs,Q.-prevQ));
    return diff
end;

# parameter and directory setting
h = ARGS[2];    # width of fitness distribution
maxtime = 1000000;
n_sp = 20;      # total number of species
n_com = 100;    # the number of local communities
nu = 0;         # immigration rate
aid = parse(Int64,ARGS[1]); # for parallel computing with different parameters from Parameters(aid).
topology,autocorr,td,w_adjust = Parameters(aid);

# output directly specification and file name settings
cd(string("path-to-directory-for-output"));
filename = string("connectivity-file-name"); # e.g. Complete_undir
envFile = string("environmental-condition-file-name"); # e.g. Complete_Lowest
name = envFile;

pinit = 1 ./ n_sp;

A = ReadMat(filename,w_adjust);
E_ls = ReadEnvs(envFile);
C = ReproductionRate(n_sp,n_com,parse(Float64,h),E_ls);
Q = Initial(n_sp,n_com,pinit);
Preg = vec(sum(Q,dims=2) ./ Float64(sum(Q)));
adiv = zeros(0);
bdiv = zeros(0);
gdiv = zeros(0);
tolerance = 0.0000001 .* n_sp .* n_com;
Qdiff = zeros(0);
biomass = zeros(0);
g = 0;
while g < maxtime
    global prevQ = deepcopy(Q); # update the prevQ using the previous Q
    global Q = Update(Q,A,C,nu,Preg,n_com); # new Q

    diff = calcDiff(Q,prevQ);
    append!(Qdiff,diff);
    append!(biomass,sum(Q));
    if Qdiff[end] <= tolerance
        break
    end
    global g += 1;
end
if g == maxtime
    println("runtime error")
end
writedlm(string("Q_",name,".txt"),Q,' ');
writedlm(string("C_",name,".txt"),C,' ');
writedlm(string("Qdiff_",name,".txt"),Qdiff,' ');
