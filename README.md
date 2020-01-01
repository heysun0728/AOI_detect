# AOI_detect

## 專案目標
參與AIdea網站的AOI自動光學檢測競賽


## 使用方式
1.	至AIdea網站下載訓練用資料 [link](https://aidea-web.tw/topic/a49e3f76-69c9-4a4a-bcfc-c882840b3f27)

2.	資料集整理
    *  至train_images資料夾內，新增三個資料夾test、train、valid，並在此三個資料夾內分別新增名字為0、1、2、3、4、5的資料夾
    * 執行 `python initial.py` 將圖片按照7:1.5:1.5的比例分配至test、train、valid三個資料夾內，並且按照圖片的分類放置0、1、2、3、4、5的資料夾內

3.	程式執行
執行 `python final.py`，來運行模型訓練，最終會印出使用train_images/test內測試資料進行測試的結果，倘若結果不錯可使用下一步驟產生AIdea需要的csv檔案上傳評分。

4.	輸出AIdea要求的csv檔
執行 `python predict.py`，除了再次印出使用測試資料進行測試的結果外，也會在資料夾內產生output.csv的檔案，倘若此檔已存在即立刻複寫。
