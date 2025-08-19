### call the package
library(BGLR)
library(AGHmatrix)

# upload the markers in your R environment Numerical_input_filtered_imputed_EYT_22_23

X<-as.matrix(read.table(file="markers-file-name.txt", header = T, row.names = 1, check.names = F))
print(X[1:5, 1:5])
#check the dimension
dim(X)

# upload the phenotype in your R environment
pheno<-read.csv(file = "pheno_file_name.csv", header = T)
pheno$GID<-as.character(pheno$GID)
#check the dimension
dim(pheno)
head(pheno)

### when you performed the filtering, probably some genotypes have been discarded
#so we have use the same genotypes for SNP and pheno information
remain_GID<-rownames(X) #identify which ones are the remaining genotypes from the marker filtering process
pheno <- pheno[as.character(pheno$GID) %in% remain_GID, ] #this line is for selecting the remaining genotypes from your phenotipic dataset, the you should to have the name of the lines as rownames.

remaing_pheno<-pheno$GID

X <- X[as.character(rownames(X)) %in% remaing_pheno, ] #this line is for selecting the remaining genotypes from your phenotipic dataset, the you should to have the name of the lines as rownames.
dim(X)

#G matrix estimation
X<-as.matrix(X)
G<-Gmatrix(X,method="VanRaden")
G <- G[order(rownames(G)), order(colnames(G))] # order the G matrix by column and row name

pheno<-pheno[order(pheno$GID),]
# Check alignment
stopifnot(all(rownames(G) == pheno$GID))

#check if there is correspondence between G and pheno files
head(pheno$GID)
head(rownames(G))
head(colnames(G))


tail(pheno$GID)
tail(rownames(G))
tail(colnames(G))


print(G[c(1:10), c(1:10)])


colnames(pheno)


############### CV

#index<-pheno[,c(1,6,10,18,22,26,14,39:48)]
#waves<-pheno[,c(1,5,9,17,21,25)]
#waves<-pheno[,c(1,6,10,18,22,26)]
#index<-pheno[,c(1,5,9,17,21,25,13,29:38)]
#index<-pheno[,c(1,5,9,17,21,25,13,29:38)]

rownames(waves)<-waves[,1]
waves<-waves[,-1]
colnames(waves)

rownames(index)<-index[,1]
index<-index[,-1]
colnames(index)


#build the H matrix
X <- scale(index, scale = T) #scale the markers
H <- tcrossprod(X)/ncol(X) # perform the crossproduct of the scaled markers
H <- H[order(rownames(H)), order(colnames(H))] # order the G matrix by column and row name


#check if there is correspondence between G and pheno files
head(pheno$GID)
head(rownames(H))
head(colnames(H))


tail(pheno$GID)
tail(rownames(H))
tail(colnames(H))


# Here we define our predictors, we want to use the relationship across the individuals (G) for make predictions
#ETA<-list(list(K=G,model="RKHS"),
#          list(K=H,model="RKHS"))

#ETA<-list(list(K=G,model="RKHS"))
#ETA<-list(list(K=H,model="RKHS"))
# create the empty matrix
cor_matrix <- matrix(NA, nrow = 5, ncol = 10)  # Assuming you have 5 folds and you want to repeat the cross validation for 10 times (you can change this)


# Repeat the cross-validation process 10 times, you can change if you want to repeat the loop five times, you can just put 1:5
for (iter in 1:10) {
  # Set up variables
  y <- pheno$score_20240809 # assuming GY is the name of your trait
  n <- length(y) # to calculate the length of the genotypes we are going to analyse
  folds <- sample(1:5, size = n, replace = TRUE) # based on the size of our population, we split randomly our population in 5 parts
  
  # here the loop for each parts of the fifth we have.
  for (i in 1:max(folds)) {
    tst <- which(folds == i) #in this case we are taking a part as testing population
    yNA <- y
    yNA[tst] <- NA #we put NAs for this population because we want to predict this part
    #this is the model, we use the pgenotype in y, yNA is the vector of the target variables with the NAs for the testing population,
    # ETA are our predictors. nIter is the number of the interactions,this is how many times the model has an output and this oputput is used as input again, this number is usually related to tha size of the population (12k is okay).
    # burnIn is the number of the interaction which the model use for warming up (5k is okay).
    fm <- BGLR(y = yNA, ETA = ETA, nIter = 12000, burnIn = 5000)
    yp_ts <- fm$yHat # fm is a list which has several parameters. you can find the prediction as yHat
    cor_matrix[i, iter] <- cor(y[tst], yp_ts[tst], use = "complete") # we are interested in make the correlation between our observed values and the prediction of each test population.so we use the function cor for the correlations which shoudl be just for the testing [tst]. we store all the correlations in the matrix cor_matrix
  }
}

summary(cor_matrix)

write.csv(cor_matrix, file = "name_your_output.csv")

