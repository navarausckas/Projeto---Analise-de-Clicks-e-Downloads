#Configurando o diretorio de trabalho
setwd("~/Desktop/Curso_1 _BigDataAnalytics_R_e_Azure/CAP20_Projetos_FeedBack/Projeto1")
getwd()

# Pacotes e bibliotecas utilizados
library("data.table")
library("dplyr")
library("lubridate")
library("ggplot2")
library("caTools")
library("randomForest")
library("e1071")
library("caret")
library("neuralnet")
library("class")
library("gmodels")


# 1 - Coleta de Dados
# Devido aos datasets serem muito grandes, vou utilizar o arquivo train.csv com 50 milhoes de observacoes
system.time(dados_provisorios <- fread("train.csv", nrows=50000000))


# 2 - Analise exploratória inicial dos dados
head(dados_provisorios)
tail(dados_provisorios)
View(dados_provisorios)
class(dados_provisorios)
dim(dados_provisorios)
str(dados_provisorios)

# Avaliar se os dados estão balanceados
table(dados_provisorios$is_attributed)

# Conforme constatado no item anterior os dados estao muito desbalanceados
# 131.233 para o campo is_attributed = 1
# 49.868.767 para o campo is_attributed = 0
# Portanto, irei fazer uma amostragem de 100.000 para 0 e 1 dessa forma teremos um 
# novo dataset balanceado com 50% para ambos.

click_false <- dados_provisorios %>% 
  filter(is_attributed == 0)%>%
  sample_n(size = 100000)

click_true <- dados_provisorios %>% 
  filter(is_attributed == 1)%>%
  sample_n(size = 100000)

# Unindo as amostras e salvando em arquivo
dados <- rbind(click_false,click_true)
class(dados)
any(is.na(dados))

# Observe que agora temos dados balanceados
table(dados$is_attributed)

# Salvando a primeira versao do arquivo
write.csv(dados, "dados_v1.csv")


# 3 - Analise, tratamento e transformação dos dados
head(dados)
str(dados)
View(dados)

# Alterando nome das variaveis
colnames(dados) <- c('ip','app','device','os','channel','click_time','download_time','download')

# Alterando a variavel target para os valores 0 e 1 para Nao e Sim e alterando para fator
dados$download = sapply(dados$download, function(x){ifelse(x==0, 'Nao', 'Sim')})
dados$download <- factor(dados$download, levels = c("Sim", "Nao"), labels = c("Sim", "Nao"))


# Convertendo os dados para formato POSIXct
dados$click_time <- ymd_hms(dados$click_time)
dados$download_time <- ymd_hms(dados$download_time)

# Incluindo duas novas colunas (Hora Clique e Hora Downloado)
dados$hora_download <- format(dados$download_time, "%T")
dados$hora_click <- format(dados$click_time, "%T")

# Incluindo e convertendo para numerico variaveis para analisar o periodo de clicks
dados$periodo_download <- as.integer(format(dados$download_time, "%H"))
dados$periodo_click <- as.integer(format(dados$click_time, "%H"))
dados$hora_cheia <- as.integer(format(dados$click_time, "%H"))

dados$periodo_click = sapply(dados$periodo_click, function(x){
  ifelse(x >= 0 & x<=6, "Madrugada", 
         ifelse(x > 6 & x<= 12, "Manha",    
                ifelse(x > 12 & x<= 18, "Tarde", "Noite")))})

dados$periodo_download = sapply(dados$periodo_download, function(x){
  ifelse(x >= 0 & x<=6, "Madrugada", 
         ifelse(x > 6 & x<= 12, "Manha",    
                ifelse(x > 12 & x<= 18, "Tarde", "Noite")))})


# Criando fator da variavel periodo_click para analise dos dados
dados$periodo_click <- factor(dados$periodo_click, levels = c("Madrugada", "Manha", "Tarde", "Noite"), labels = c("Madrugada", "Manha", "Tarde", "Noite"))


# Analise exploratoria para identificar informacoes por periodos e horarios
dados %>% 
  group_by(periodo_click) %>%
  filter(download == "Nao") %>%
  summarise(total = n())

dados %>% 
  group_by(periodo_download) %>%
  filter(download == "Sim") %>%
  summarise(total = n())

dados %>% 
  group_by(periodo_click) %>%
  filter(download == "Sim") %>%
  summarise(total = n()) %>%
  ggplot(aes(x = periodo_click, y = total)) +
  geom_bar(stat = "identity") 


dados %>% 
  group_by(periodo_download) %>%
  filter(download == "Sim") %>%
  summarise(total = n()) %>%
  ggplot(aes(x = periodo_download, y = total)) +
  geom_bar(stat = "identity")


# Grafico comparativo de downloads por periodo
dados %>% 
  group_by(periodo_click, download) %>%
  summarise(total = n()) %>%
  ggplot(aes(x = periodo_click, y = total, fill = download)) +
  geom_bar(stat = "identity", position=position_dodge()) 


# Grafico comparativo de downloads por hora
dados %>% 
  group_by(hora_cheia, download) %>%
  summarise(total = n()) %>%
  ggplot(aes(x = hora_cheia, y = total, fill = download, color = download)) +
  #geom_bar(stat = "identity", position=position_dodge()) 
  geom_line()+
  geom_point()






# 4 - FeatureEngineering

# Transformando variáveis numéricas em variáveis categóricas
max(dados$app)
max(dados$device)
max(dados$os)
max(dados$channel)

cut_factor <- function(x, nlevs = 5, maxval = 3600, #nlevs = niveis de categoria
                         minval = 0, ordered = TRUE){
  cuts <- seq(min(x), max(x), length.out = nlevs + 1)
  cuts[1] <- minval
  cuts[nlevs + 1] <- maxval
  print(cuts)
  x <- cut(x, breaks = cuts, right = FALSE, order_result = ordered) 
}

# Incluindo novas colunas do tipo fator
dados$app_f <- cut_factor(dados$app)
dados$device_f <- cut_factor(dados$device)
dados$os_f <- cut_factor(dados$os)
dados$channel_f <- cut_factor(dados$channel)

# Checando se há valores missing
any(is.na(dados$app_f))
any(is.na(dados$device_f))
any(is.na(dados$os_f))
any(is.na(dados$channel_f))

# Gravando os dados alterados em um novo arquivo
# Salvando a segunda versao do arquivo
write.csv(dados, "dados_v2.csv")

str(dados)
head(dados)

#system.time(dados <- fread("dados_v2.csv"))
#dados$X = NULL
dados <- read.csv("dados_v2.csv", header = TRUE)

# Criando dados de treino e de teste (60% e 40% respectivamente)
amostra <- sample.split(dados$download, SplitRatio = 0.60)
treino = subset(dados, amostra == TRUE)
teste = subset(dados, amostra == FALSE)

# Verificando se os dados ficaram balanceados
table(treino$download)
table(teste$download)



#5 Feature Selection e testes de modelos

# Modelo randomForest para criar um plot de importância das variáveis
# Algumas variaveis que foram criadas para analise exploratoria possuem valores NA e nao iremos
# considera-las para a criação do modelo.
any(is.na(dados$hora_click))
any(is.na(dados$periodo_click))
any(is.na(dados$hora_cheia))

modelo <- randomForest(download ~  
                        + ip
                        + app 
                        + device 
                        + os
                        + channel
                        + click_time
                        #+ hora_click  
                        #+ periodo_click
                        + hora_cheia
                        + app_f
                        + device_f
                        + os_f
                        + channel_f,
                        #- download_time
                        #- hora_download
                        #- periodo_download,
                        data = treino,
                        ntree = 100, nodesize = 10, importance = T)

# Observamos tambem que algumas variaveis que criamos fatores com a funcao CUT
# não geram um impacto significativo para criacao do modelo, sendo assim iremos retira-las.
varImpPlot(modelo)


#Modelo 0 Random Forest
modelo_v0_rf <- randomForest( download ~ + app + ip + channel + click_time + hora_cheia + channel_f + device,
                              data = treino,
                              ntree = 100, nodesize = 10)

print(modelo_v0_rf)

# Ao observar a acuracia do modelo criado com as variaveis fatores criadas, 
# foi possivel observar que o modelo nao foi bom... aproximadamente 91.5% de acuracia
previsoes_v0 <- predict(modelo_v0_rf, teste, type = "class")

## Percentual de previsões corretas com dataset de teste
mean(previsoes_v0==teste$download)

# Confusion Matrix
table(previsoes_v0, teste$download)



#Modelo 1 Random Forest
modelo_v1_rf <- randomForest( download ~ + channel_f + app_f + device_f + os_f + click_time,
                              data = treino,
                              ntree = 100, nodesize = 10)


print(modelo_v1_rf)
# Ao observar a acuracia do modelo criado com as variaveis fatores criadas, 
# foi possivel observar que o modelo nao foi bom... aproximadamente 64% de acuracia
previsoes_v1 <- predict(modelo_v1_rf, teste, type = "class")

## Percentual de previsões corretas com dataset de teste
mean(previsoes_v1==teste$download)

# Confusion Matrix
table(previsoes_v1, teste$download)



# Modelo 2 Random Forest
# Ao observar este novo modelo o mesmo apresentou uma acuracia de aproximadamente 92%
modelo_v2_rf <- randomForest( download ~ + ip + app + device + os + channel,
                              data = treino,
                              ntree = 100, nodesize = 10)

print(modelo_v2_rf)

# Ao observar a acuracia do modelo criado com as variaveis fatores criadas, 
# foi possivel observar que o modelo nao foi bom... aproximadamente 64% de acuracia

previsoes_v2 <- predict(modelo_v2_rf, teste, type = "class")

## Percentual de previsões corretas com dataset de teste
mean(previsoes_v2==teste$download)

# Confusion Matrix
table(previsoes_v2, teste$download)




# Agora vou fazer a normalizacao dos dados para avaliar se há melhorias no modelo e testar
# com outros modelos alem do Random Forest.

# Vou retirar algumas variaveis deixando somente aquelas mais importantes
dados$click_time <- NULL
dados$download_time <- NULL
dados$hora_click <- NULL
dados$hora_cheia <- NULL
dados$periodo_click <- NULL
dados$hora_download <- NULL
dados$periodo_download <- NULL
dados$app_f <- NULL
dados$device_f <- NULL
dados$channel_f <- NULL
dados$os_f <- NULL

str(dados)

# Funcao para normalizar os dados
normalizar <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Normalizando dados
download <- dados$download
head(download)

dados$download = NULL

dados <- as.data.frame(lapply(dados[ , 1:5], normalizar))
dados <- cbind(dados,download)

head(dados)
str(dados)
class(dados)

# Gravando os dados alterados em um novo arquivo
# Salvando a terceira versao do arquivo com dados normalizados
write.csv(dados, "dados_v3.csv")

# Criando novos dados de treino e de teste (60% e 40% respectivamente)
amostra <- sample.split(dados$download, SplitRatio = 0.60)
treino = subset(dados, amostra == TRUE)
teste = subset(dados, amostra == FALSE)

# Verificando se os dados ficaram balanceados
table(treino$download)
table(teste$download)


# Modelo 3 Random Forest (com dados normalizados)
# Ao observar este novo modelo o mesmo apresentou uma acuracia de aproximadamente 92%

modelo_v3_rf <- randomForest(download ~ .,
                            data = treino,
                            ntree = 100, nodesize = 10)

print(modelo_v3_rf)

previsoes_v3 <- predict(modelo_v3_rf, teste, type = "class")

## Percentual de previsões corretas com dataset de teste
mean(previsoes_v3==teste$download)

# Confusion Matrix
table(previsoes_v3, teste$download)




# Utilizando SVM
# Aproximadamente 82% de acuracia
modelo_svm_v1 <- svm(download ~ ., 
                     data = treino, 
                     type = 'C-classification', 
                     kernel = 'radial') 

# Previsões nos dados de teste
previsoes_svm <- predict(modelo_svm_v1, teste) 

# Percentual de previsões corretas com dataset de teste
mean(previsoes_svm == teste$download)  

# Confusion Matrix
table(previsoes_svm, teste$download)




# Utilizando Naive Bayes
# Aproximadamente 70% de acuracia
modelo_naive_bayes <- naiveBayes(download ~ ., data = treino) 

# Fazendo as Previsões
previsoes_nb <- predict(modelo_naive_bayes, teste)

# Média
mean(previsoes_nb == teste$download)

# Confusion matrix
table(pred = previsoes_nb, true = teste$download)



# Utilizando Caret
# Acuracia de aproximadamente 86%

modelo_caret <- train(download ~ ., data = treino, method = "knn")
varImp(modelo_caret)

# Resumo do modelo
summary(modelo_caret)

# Previsoes
previcoes_caret <- predict(modelo_caret, teste)
previcoes_caret
plot(teste$download, previcoes_caret)

#Media
mean(previcoes_caret == teste$download)  

# Confusion Matrix
table(previcoes_caret, teste$download)





# Utilizando KNN
# Acuracia de aproximadamente 88%

# Criando os labels para os dados de treino e de teste
dados_treino_labels <- treino[ , 6] 
dados_teste_labels <- teste[ , 6]

length(dados_treino_labels)
length(dados_teste_labels)

# Retirando variavel download
treino$download <- NULL
teste$download <- NULL


# Criando o modelo
modelo_knn_v1 <- knn(train = treino, 
                     test = teste,
                     cl = dados_treino_labels, 
                     k = 21)

# Confusion Matrix
CrossTable(x = dados_teste_labels, y = modelo_knn_v1, prop.chisq = FALSE)


################## CONCLUSAO ###################
# Com base nos testes e analises realizadas constatamos que a melhor opcao
# para a criação do modelo neste caso é utilizando Random Forest 
# modelo_v3_rf que obeteve uma acuracia de aproximadamente 92%








