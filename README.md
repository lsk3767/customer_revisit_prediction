# Patient Revisit Prediction System

## 프로젝트 개요

의료 데이터를 기반으로 환자의 재방문 여부를 예측하는 머신러닝 시스템

---

## 데이터 구조

* mdcl_info (진료 정보)
* rcpt_info (수납 정보)
* cust_info (고객 정보)

---

## Feature Engineering

* gap_days: 방문 간격
* visit_count: 총 방문 횟수
* days_since_visit: 마지막 방문 이후 경과일

---

## Label

* 90일 이내 재방문 여부 (1: 재방문 / 0: 이탈)

---

## 모델

* Logistic Regression

---

## 현재 성능

* Accuracy: 0.70
* AUC: 0.73
* 문제: 클래스 불균형으로 인해 이탈 환자 예측 성능 낮음

---

## 시스템 구조

Node.js → Python → DB 업데이트 자동화 진행 중

---

## 향후 개선 계획

* 클래스 불균형 해결
* Feature 추가
* 모델 고도화 (XGBoost, LightGBM)
