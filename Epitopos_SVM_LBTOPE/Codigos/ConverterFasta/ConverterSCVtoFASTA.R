################################################################################
################################################################################
################################################################################
##Converter sequências csv para FASTA
##Tiago Tambonis.
################################################################################
################################################################################
################################################################################

################################################################################
##Bibliotecas
################################################################################
library("seqinr")
library("protr")

nomepositivo = "LBtope_Fixed_non_redundant_Positive_pattern.txt"
nomenegativo = "LBtope_Fixed_non_redundant_Negative_pattern.txt"

#Positives
im_seq <- read.table(nomepositivo , header = FALSE, stringsAsFactors = FALSE)
im_seq <- sapply(im_seq, print)
im_seq <- as.list(im_seq)

#Negatives
non_im_seq <- read.table(nomenegativo, header = FALSE, stringsAsFactors = FALSE)
non_im_seq <- sapply(non_im_seq, print)
non_im_seq <- as.list(non_im_seq)

################################################################################
##Filtrar sequências iguais.
################################################################################
if (FALSE){
  im_seq_temp = im_seq[!(im_seq %in% non_im_seq)]
  non_im_seq = non_im_seq[!(non_im_seq %in% im_seq)]
  im_seq = im_seq_temp
  rm(im_seq_temp)}

#Salvar arquivos
write.fasta(sequences = non_im_seq, names = paste("Seq_", seq(length(non_im_seq)), sep=""), 
            nbchar = 80, file.out =  paste(nomenegativo, ".fasta", sep=""), as.string=TRUE)

write.fasta(sequences = im_seq, names = paste("Seq_", seq(length(im_seq)), sep=""), nbchar = 80, 
            file.out =  paste(nomepositivo, ".fasta", sep=""), as.string=TRUE)