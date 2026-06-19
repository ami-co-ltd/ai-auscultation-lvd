library(tidyverse)
library(pROC)
library(ggplot2)
library(tableone)
library(readxl)
library(writexl)
library(boot)
set.seed(0)
here::here()

internal_data_raw <- read.csv("BNP_imputed_internal_mean.csv")

# 1. 欠測のある行を除いた解析用データを作成（使う変数をすべて指定）
internal_data <- internal_data_raw %>%
  filter(!is.na(LVEF2_cat) & !is.na(age) & !is.na(sex) & !is.na(BMI) & !is.na(DPC_IHD) & !is.na(DPC_CM) & !is.na(Afib) & !is.na(BNP))
# 対数BNPを算出
internal_data_raw <- internal_data_raw %>%
  mutate(log_BNP = log(BNP))

# 2. 欠測除外後の症例IDを確認
excluded_ids_internal <- setdiff(internal_data_raw$ID, internal_data$ID)

# 3. glm 実行
regression_model <- glm(LVEF2_cat ~ age + sex + BMI + DPC_IHD + DPC_CM + Afib + log_BNP, data = internal_data_raw, family = binomial)

# 4. 予測確率
internal_data <- internal_data %>%
  mutate(pred_prob = predict(regression_model, type = "response"))

# 5. ROCとAUC
roc_reg_model <- roc(response = internal_data$LVEF2_cat, predictor = internal_data$pred_prob)
plot(roc_reg_model, print.auc = TRUE)
Youden_index_model <- coords(roc_reg_model, "best", best.method = "youden", ret = "threshold")
Youden_threshold <- Youden_index_model$threshold
# AUROCのブートストラップによる95%信頼区間を取得
auc_ci_reg_model <- ci(
  roc_reg_model,
  method = "bootstrap",
  boot.n = 2000,
  progress = "none"  # ワーニング防止
)

print(auc_ci_reg_model)

# 6. Listen Externalに当てはめる#########################################################################################################################################################
external_data_raw <- read.csv("BNP_imputed_external_mean.csv")

external_data <- external_data_raw %>% 
  filter(!is.na(labels) & !is.na(age) & !is.na(sex) & !is.na(BMI) & !is.na(DPC_IHD) & !is.na(DPC_CM) & !is.na(Afib) & !is.na(BNP))
external_data <- external_data %>%
  mutate(log_BNP = log(BNP))

external_data <- external_data %>% 
  mutate(pred_prob = predict(regression_model, newdata = external_data, type = "response"))

# 7. internalのYouden Indexを当てはめる
external_data <- external_data %>% 
  mutate(pred_bin = if_else(pred_prob >= Youden_threshold, 1, 0))

# 8. 性能評価
roc_reg_model_external <- roc(external_data$LVEF2_cat, external_data$pred_prob)
plot(roc_reg_model_external, print.auc =TRUE)

# AUROCのブートストラップによる95%信頼区間を取得
auc_ci_reg_model_external <- ci(
  roc_reg_model_external,
  method = "bootstrap",
  boot.n = 2000,
  progress = "none"
)

# 表示
print(auc_ci_reg_model_external)


# McFaddenの疑似決定係数
feature_cols <- c("log_BNP", "age", "sex", "BMI", "DPC_IHD", "DPC_CM", "Afib")

# モデル全体の擬似R²
model_full <- glm(LVEF2_cat ~ log_BNP + age + sex + BMI + DPC_IHD + DPC_CM + Afib,
                  data = internal_data_raw, family = binomial())
model_null <- glm(LVEF2_cat ~ 1, data = internal_data_raw, family = binomial())

pseudo_r2 <- 1 - logLik(model_full) / logLik(model_null)
cat("McFadden R²:", round(pseudo_r2, 4), "\n")

sapply(feature_cols, function(col) {
  remaining <- setdiff(feature_cols, col)
  formula_reduced <- as.formula(paste("LVEF2_cat ~", paste(remaining, collapse = " + ")))
  model_reduced <- glm(formula_reduced, data = internal_data_raw, family = binomial())
  
  r2_full    <- as.numeric(1 - logLik(model_full) / logLik(model_null))
  r2_reduced <- as.numeric(1 - logLik(model_reduced) / logLik(model_null))
  r2_full - r2_reduced  # その変数を除いたときのR²の低下量
})

summary(model_full)

# 単変量AUROC
sapply(feature_cols, function(col) {
  formula <- as.formula(paste("LVEF2_cat ~", col))
  model <- glm(formula, data = internal_data_raw, family = binomial())
  pred <- predict(model, newdata = external_data, type = "response")
  roc_obj <- pROC::roc(external_data$LVEF2_cat, pred, quiet = TRUE)
  as.numeric(pROC::auc(roc_obj))
  
  ci_result <- ci(
    roc_obj,
    method = "bootstrap",
    boot.n = 2000
  )
})

