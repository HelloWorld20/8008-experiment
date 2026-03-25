"""shell
开始时间序列预测分析...
正在加载数据...
使用本地数据文件
为测试期间添加零销售数据...
优化内存使用...
优化前内存使用: sales 454.5MB, calendar 0.2MB, prices 208.8MB
优化后内存使用: sales 96.9MB, calendar 0.1MB, prices 130.5MB
正在转换数据格式...
正在合并数据...
优化合并后数据的内存使用...

=== 开始EDA分析 ===

=== 开始特征工程 ===
准备特征工程...
转换列为分类类型...
存储类别映射...
清理内存...
执行标签编码...
编码前的数据类型:
id category
item_id category
dept_id category
cat_id category
store_id category
state_id category
d str
sold int16
date str
wm_yr_wk int16
weekday str
wday int8
month int8
year int16
event_name_1 str
event_type_1 str
event_name_2 str
event_type_2 str
snap_CA int8
snap_TX int8
snap_WI int8
sell_price float16
dtype: object
编码字符串列: weekday
编码字符串列: event_name_1
编码字符串列: event_type_1
编码字符串列: event_name_2
编码字符串列: event_type_2
删除列: date
编码分类列: id
编码分类列: item_id
编码分类列: dept_id
编码分类列: cat_id
编码分类列: store_id
编码分类列: state_id
编码后的数据类型:
id int16
item_id int16
dept_id int8
cat_id int8
store_id int8
state_id int8
d int16
sold int16
wm_yr_wk int16
weekday int8
wday int8
month int8
year int16
event_name_1 int8
event_type_1 int8
event_name_2 int8
event_type_2 int8
snap_CA int8
snap_TX int8
snap_WI int8
sell_price float16
dtype: object

=== 开始建模和预测 ===
训练LightGBM模型...
**\*** 商店预测: CA_1 **\***
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.094991 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 1350
[LightGBM] [Info] Number of data points in the train set: 5832737, number of used features: 18
[LightGBM] [Info] Start training from score 1.319829
Training until validation scores don't improve for 20 rounds
[20] training's rmse: 3.1749 training's l2: 10.08 valid_1's rmse: 2.8329 valid_1's l2: 8.02532
[40] training's rmse: 3.08081 training's l2: 9.49137 valid_1's rmse: 2.75008 valid_1's l2: 7.56296
[60] training's rmse: 2.99983 training's l2: 8.99896 valid_1's rmse: 2.69996 valid_1's l2: 7.28979
[80] training's rmse: 2.91266 training's l2: 8.48361 valid_1's rmse: 2.6336 valid_1's l2: 6.93587
[100] training's rmse: 2.86101 training's l2: 8.18538 valid_1's rmse: 2.59515 valid_1's l2: 6.73479
[120] training's rmse: 2.83011 training's l2: 8.00954 valid_1's rmse: 2.56951 valid_1's l2: 6.60238
[140] training's rmse: 2.81807 training's l2: 7.94151 valid_1's rmse: 2.55739 valid_1's l2: 6.54023
[160] training's rmse: 2.7927 training's l2: 7.7992 valid_1's rmse: 2.5447 valid_1's l2: 6.47548
[180] training's rmse: 2.77533 training's l2: 7.70246 valid_1's rmse: 2.53213 valid_1's l2: 6.4117
[200] training's rmse: 2.74892 training's l2: 7.55656 valid_1's rmse: 2.51943 valid_1's l2: 6.34754
[220] training's rmse: 2.73647 training's l2: 7.48827 valid_1's rmse: 2.51216 valid_1's l2: 6.31095
[240] training's rmse: 2.72226 training's l2: 7.41069 valid_1's rmse: 2.50448 valid_1's l2: 6.2724
[260] training's rmse: 2.70739 training's l2: 7.32994 valid_1's rmse: 2.49868 valid_1's l2: 6.24341
[280] training's rmse: 2.69798 training's l2: 7.27911 valid_1's rmse: 2.491 valid_1's l2: 6.20508
[300] training's rmse: 2.68448 training's l2: 7.20644 valid_1's rmse: 2.47933 valid_1's l2: 6.14708
[320] training's rmse: 2.67365 training's l2: 7.14843 valid_1's rmse: 2.47034 valid_1's l2: 6.10259
[340] training's rmse: 2.66809 training's l2: 7.11871 valid_1's rmse: 2.46631 valid_1's l2: 6.08267
[360] training's rmse: 2.66199 training's l2: 7.08617 valid_1's rmse: 2.46519 valid_1's l2: 6.07717
[380] training's rmse: 2.65563 training's l2: 7.05237 valid_1's rmse: 2.46208 valid_1's l2: 6.06183
[400] training's rmse: 2.6535 training's l2: 7.04104 valid_1's rmse: 2.46 valid_1's l2: 6.05162
[420] training's rmse: 2.64696 training's l2: 7.00638 valid_1's rmse: 2.45737 valid_1's l2: 6.03865
Early stopping, best iteration is:
[417] training's rmse: 2.64805 training's l2: 7.01215 valid_1's rmse: 2.45717 valid_1's l2: 6.03769
**\*** 商店预测: CA_2 **\***
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.106278 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 1352
[LightGBM] [Info] Number of data points in the train set: 5832737, number of used features: 19
[LightGBM] [Info] Start training from score 0.974753
Training until validation scores don't improve for 20 rounds
[20] training's rmse: 2.23212 training's l2: 4.98236 valid_1's rmse: 2.60875 valid_1's l2: 6.80557
[40] training's rmse: 2.16525 training's l2: 4.68833 valid_1's rmse: 2.53328 valid_1's l2: 6.4175
[60] training's rmse: 2.12114 training's l2: 4.49925 valid_1's rmse: 2.49621 valid_1's l2: 6.23107
[80] training's rmse: 2.09305 training's l2: 4.38087 valid_1's rmse: 2.47351 valid_1's l2: 6.11825
[100] training's rmse: 2.07353 training's l2: 4.29954 valid_1's rmse: 2.44414 valid_1's l2: 5.97381
[120] training's rmse: 2.04737 training's l2: 4.19173 valid_1's rmse: 2.40977 valid_1's l2: 5.80697
[140] training's rmse: 2.02884 training's l2: 4.1162 valid_1's rmse: 2.37252 valid_1's l2: 5.62884
[160] training's rmse: 2.01694 training's l2: 4.06805 valid_1's rmse: 2.35427 valid_1's l2: 5.5426
[180] training's rmse: 2.00378 training's l2: 4.01515 valid_1's rmse: 2.33666 valid_1's l2: 5.45998
[200] training's rmse: 1.9941 training's l2: 3.97643 valid_1's rmse: 2.32531 valid_1's l2: 5.40707
[220] training's rmse: 1.98514 training's l2: 3.94079 valid_1's rmse: 2.30678 valid_1's l2: 5.32123
[240] training's rmse: 1.97679 training's l2: 3.90771 valid_1's rmse: 2.29605 valid_1's l2: 5.27185
[260] training's rmse: 1.96949 training's l2: 3.87889 valid_1's rmse: 2.28711 valid_1's l2: 5.23087
[280] training's rmse: 1.95938 training's l2: 3.83917 valid_1's rmse: 2.27497 valid_1's l2: 5.17551
[300] training's rmse: 1.95201 training's l2: 3.81034 valid_1's rmse: 2.27138 valid_1's l2: 5.15917
[320] training's rmse: 1.9465 training's l2: 3.78885 valid_1's rmse: 2.2679 valid_1's l2: 5.14338
[340] training's rmse: 1.94288 training's l2: 3.77478 valid_1's rmse: 2.26241 valid_1's l2: 5.11848
[360] training's rmse: 1.93608 training's l2: 3.74839 valid_1's rmse: 2.25883 valid_1's l2: 5.10232
[380] training's rmse: 1.93095 training's l2: 3.72856 valid_1's rmse: 2.25278 valid_1's l2: 5.075
[400] training's rmse: 1.92493 training's l2: 3.70535 valid_1's rmse: 2.24581 valid_1's l2: 5.04365
[420] training's rmse: 1.92127 training's l2: 3.69127 valid_1's rmse: 2.24168 valid_1's l2: 5.02512
[440] training's rmse: 1.9177 training's l2: 3.67756 valid_1's rmse: 2.23734 valid_1's l2: 5.00569
[460] training's rmse: 1.91398 training's l2: 3.66332 valid_1's rmse: 2.23464 valid_1's l2: 4.99363
[480] training's rmse: 1.91009 training's l2: 3.64843 valid_1's rmse: 2.23111 valid_1's l2: 4.97785
[500] training's rmse: 1.90692 training's l2: 3.63635 valid_1's rmse: 2.2302 valid_1's l2: 4.9738
[520] training's rmse: 1.90379 training's l2: 3.62443 valid_1's rmse: 2.2269 valid_1's l2: 4.9591
[540] training's rmse: 1.90035 training's l2: 3.61134 valid_1's rmse: 2.22151 valid_1's l2: 4.93509
[560] training's rmse: 1.89751 training's l2: 3.60054 valid_1's rmse: 2.21773 valid_1's l2: 4.91831
[580] training's rmse: 1.89507 training's l2: 3.59128 valid_1's rmse: 2.21722 valid_1's l2: 4.91609
[600] training's rmse: 1.89231 training's l2: 3.58083 valid_1's rmse: 2.2159 valid_1's l2: 4.91023
[620] training's rmse: 1.88798 training's l2: 3.56449 valid_1's rmse: 2.21041 valid_1's l2: 4.88589
[640] training's rmse: 1.88536 training's l2: 3.55457 valid_1's rmse: 2.20755 valid_1's l2: 4.87328
[660] training's rmse: 1.88287 training's l2: 3.5452 valid_1's rmse: 2.20768 valid_1's l2: 4.87386
[680] training's rmse: 1.8804 training's l2: 3.5359 valid_1's rmse: 2.20483 valid_1's l2: 4.8613
[700] training's rmse: 1.87817 training's l2: 3.52754 valid_1's rmse: 2.20382 valid_1's l2: 4.85682
[720] training's rmse: 1.87628 training's l2: 3.52042 valid_1's rmse: 2.20253 valid_1's l2: 4.85114
[740] training's rmse: 1.87363 training's l2: 3.5105 valid_1's rmse: 2.20115 valid_1's l2: 4.84506
[760] training's rmse: 1.87205 training's l2: 3.50456 valid_1's rmse: 2.20016 valid_1's l2: 4.8407
[780] training's rmse: 1.87022 training's l2: 3.49774 valid_1's rmse: 2.19911 valid_1's l2: 4.8361
[800] training's rmse: 1.8677 training's l2: 3.48831 valid_1's rmse: 2.19678 valid_1's l2: 4.82584
[820] training's rmse: 1.86581 training's l2: 3.48126 valid_1's rmse: 2.19652 valid_1's l2: 4.82469
[840] training's rmse: 1.86423 training's l2: 3.47535 valid_1's rmse: 2.19388 valid_1's l2: 4.8131
[860] training's rmse: 1.86258 training's l2: 3.46919 valid_1's rmse: 2.19249 valid_1's l2: 4.80702
[880] training's rmse: 1.86094 training's l2: 3.46308 valid_1's rmse: 2.19112 valid_1's l2: 4.80099
[900] training's rmse: 1.85941 training's l2: 3.45742 valid_1's rmse: 2.18957 valid_1's l2: 4.79421
[920] training's rmse: 1.85786 training's l2: 3.45165 valid_1's rmse: 2.18814 valid_1's l2: 4.78797
[940] training's rmse: 1.85647 training's l2: 3.44649 valid_1's rmse: 2.18769 valid_1's l2: 4.78599
[960] training's rmse: 1.85524 training's l2: 3.44191 valid_1's rmse: 2.18481 valid_1's l2: 4.77341
[980] training's rmse: 1.85382 training's l2: 3.43665 valid_1's rmse: 2.1848 valid_1's l2: 4.77335
[1000] training's rmse: 1.8522 training's l2: 3.43063 valid_1's rmse: 2.18372 valid_1's l2: 4.76865
Did not meet early stopping. Best iteration is:
[998] training's rmse: 1.85235 training's l2: 3.43121 valid_1's rmse: 2.18371 valid_1's l2: 4.7686
**\*** 商店预测: CA_3 **\***
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.096006 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 1354
[LightGBM] [Info] Number of data points in the train set: 5832737, number of used features: 19
[LightGBM] [Info] Start training from score 1.918170
Training until validation scores don't improve for 20 rounds
[20] training's rmse: 4.5149 training's l2: 20.3843 valid_1's rmse: 3.52659 valid_1's l2: 12.4368
[40] training's rmse: 4.31415 training's l2: 18.6119 valid_1's rmse: 3.38202 valid_1's l2: 11.438
[60] training's rmse: 4.21346 training's l2: 17.7532 valid_1's rmse: 3.29701 valid_1's l2: 10.8703
[80] training's rmse: 4.13951 training's l2: 17.1355 valid_1's rmse: 3.25544 valid_1's l2: 10.5979
[100] training's rmse: 4.09065 training's l2: 16.7334 valid_1's rmse: 3.21241 valid_1's l2: 10.3196
[120] training's rmse: 4.0324 training's l2: 16.2602 valid_1's rmse: 3.17864 valid_1's l2: 10.1038
[140] training's rmse: 3.99062 training's l2: 15.9251 valid_1's rmse: 3.14024 valid_1's l2: 9.86112
[160] training's rmse: 3.96794 training's l2: 15.7446 valid_1's rmse: 3.12649 valid_1's l2: 9.77496
[180] training's rmse: 3.93165 training's l2: 15.4579 valid_1's rmse: 3.11312 valid_1's l2: 9.6915
[200] training's rmse: 3.89622 training's l2: 15.1805 valid_1's rmse: 3.10183 valid_1's l2: 9.62133
[220] training's rmse: 3.87309 training's l2: 15.0008 valid_1's rmse: 3.08132 valid_1's l2: 9.49456
[240] training's rmse: 3.8552 training's l2: 14.8626 valid_1's rmse: 3.07608 valid_1's l2: 9.46227
[260] training's rmse: 3.83856 training's l2: 14.7345 valid_1's rmse: 3.0669 valid_1's l2: 9.40589
[280] training's rmse: 3.81623 training's l2: 14.5636 valid_1's rmse: 3.05188 valid_1's l2: 9.31398
[300] training's rmse: 3.80292 training's l2: 14.4622 valid_1's rmse: 3.04689 valid_1's l2: 9.28352
[320] training's rmse: 3.78488 training's l2: 14.3253 valid_1's rmse: 3.04083 valid_1's l2: 9.24664
[340] training's rmse: 3.77932 training's l2: 14.2833 valid_1's rmse: 3.03441 valid_1's l2: 9.20764
[360] training's rmse: 3.76494 training's l2: 14.1748 valid_1's rmse: 3.03106 valid_1's l2: 9.1873
[380] training's rmse: 3.75593 training's l2: 14.107 valid_1's rmse: 3.0281 valid_1's l2: 9.16937
[400] training's rmse: 3.74496 training's l2: 14.0247 valid_1's rmse: 3.01884 valid_1's l2: 9.11338
[420] training's rmse: 3.73593 training's l2: 13.9572 valid_1's rmse: 3.01319 valid_1's l2: 9.07931
[440] training's rmse: 3.72485 training's l2: 13.8745 valid_1's rmse: 3.00729 valid_1's l2: 9.04379
[460] training's rmse: 3.71366 training's l2: 13.7912 valid_1's rmse: 3.00496 valid_1's l2: 9.02976
[480] training's rmse: 3.70347 training's l2: 13.7157 valid_1's rmse: 3.00003 valid_1's l2: 9.00018
[500] training's rmse: 3.69569 training's l2: 13.6581 valid_1's rmse: 2.99256 valid_1's l2: 8.95542
[520] training's rmse: 3.68887 training's l2: 13.6078 valid_1's rmse: 2.99027 valid_1's l2: 8.9417
[540] training's rmse: 3.68541 training's l2: 13.5822 valid_1's rmse: 2.9838 valid_1's l2: 8.90304
[560] training's rmse: 3.67952 training's l2: 13.5389 valid_1's rmse: 2.98234 valid_1's l2: 8.89434
[580] training's rmse: 3.67415 training's l2: 13.4994 valid_1's rmse: 2.97956 valid_1's l2: 8.8778
[600] training's rmse: 3.66786 training's l2: 13.4532 valid_1's rmse: 2.97927 valid_1's l2: 8.87608
[620] training's rmse: 3.66195 training's l2: 13.4099 valid_1's rmse: 2.97304 valid_1's l2: 8.83895
[640] training's rmse: 3.65593 training's l2: 13.3658 valid_1's rmse: 2.97225 valid_1's l2: 8.83424
[660] training's rmse: 3.65202 training's l2: 13.3373 valid_1's rmse: 2.96886 valid_1's l2: 8.81416
[680] training's rmse: 3.64741 training's l2: 13.3036 valid_1's rmse: 2.9678 valid_1's l2: 8.80784
[700] training's rmse: 3.64235 training's l2: 13.2667 valid_1's rmse: 2.96407 valid_1's l2: 8.7857
[720] training's rmse: 3.63882 training's l2: 13.241 valid_1's rmse: 2.96431 valid_1's l2: 8.78715
Early stopping, best iteration is:
[700] training's rmse: 3.64235 training's l2: 13.2667 valid_1's rmse: 2.96407 valid_1's l2: 8.7857
**\*** 商店预测: CA_4 **\***
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.133477 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 1352
[LightGBM] [Info] Number of data points in the train set: 5832737, number of used features: 19
[LightGBM] [Info] Start training from score 0.703559
Training until validation scores don't improve for 20 rounds
[20] training's rmse: 1.64528 training's l2: 2.70696 valid_1's rmse: 1.68062 valid_1's l2: 2.8245
[40] training's rmse: 1.58736 training's l2: 2.51973 valid_1's rmse: 1.6377 valid_1's l2: 2.68206
[60] training's rmse: 1.56608 training's l2: 2.45261 valid_1's rmse: 1.61952 valid_1's l2: 2.62286
[80] training's rmse: 1.54237 training's l2: 2.3789 valid_1's rmse: 1.59914 valid_1's l2: 2.55723
[100] training's rmse: 1.52385 training's l2: 2.32211 valid_1's rmse: 1.59011 valid_1's l2: 2.52846
[120] training's rmse: 1.51365 training's l2: 2.29114 valid_1's rmse: 1.58151 valid_1's l2: 2.50118
[140] training's rmse: 1.50642 training's l2: 2.2693 valid_1's rmse: 1.57637 valid_1's l2: 2.48493
[160] training's rmse: 1.50076 training's l2: 2.25227 valid_1's rmse: 1.57047 valid_1's l2: 2.46637
[180] training's rmse: 1.49163 training's l2: 2.22495 valid_1's rmse: 1.56442 valid_1's l2: 2.44742
[200] training's rmse: 1.48269 training's l2: 2.19836 valid_1's rmse: 1.56061 valid_1's l2: 2.43551
[220] training's rmse: 1.4793 training's l2: 2.18834 valid_1's rmse: 1.55632 valid_1's l2: 2.42212
[240] training's rmse: 1.47382 training's l2: 2.17216 valid_1's rmse: 1.55288 valid_1's l2: 2.41145
[260] training's rmse: 1.47031 training's l2: 2.16182 valid_1's rmse: 1.55169 valid_1's l2: 2.40773
[280] training's rmse: 1.46459 training's l2: 2.14503 valid_1's rmse: 1.54831 valid_1's l2: 2.39728
[300] training's rmse: 1.46143 training's l2: 2.13578 valid_1's rmse: 1.54655 valid_1's l2: 2.39181
[320] training's rmse: 1.45752 training's l2: 2.12436 valid_1's rmse: 1.54388 valid_1's l2: 2.38357
[340] training's rmse: 1.45491 training's l2: 2.11676 valid_1's rmse: 1.54299 valid_1's l2: 2.38083
[360] training's rmse: 1.45184 training's l2: 2.10783 valid_1's rmse: 1.54142 valid_1's l2: 2.37598
[380] training's rmse: 1.44932 training's l2: 2.10053 valid_1's rmse: 1.53935 valid_1's l2: 2.3696
[400] training's rmse: 1.4466 training's l2: 2.09266 valid_1's rmse: 1.53785 valid_1's l2: 2.365
[420] training's rmse: 1.44425 training's l2: 2.08587 valid_1's rmse: 1.53709 valid_1's l2: 2.36265
Early stopping, best iteration is:
[401] training's rmse: 1.44642 training's l2: 2.09214 valid_1's rmse: 1.53688 valid_1's l2: 2.36201
**\*** 商店预测: TX_1 **\***
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.103321 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 1365
[LightGBM] [Info] Number of data points in the train set: 5832737, number of used features: 20
[LightGBM] [Info] Start training from score 0.959291
Training until validation scores don't improve for 20 rounds
[20] training's rmse: 2.49781 training's l2: 6.23904 valid_1's rmse: 2.42707 valid_1's l2: 5.89067
[40] training's rmse: 2.3977 training's l2: 5.74896 valid_1's rmse: 2.27627 valid_1's l2: 5.18142
[60] training's rmse: 2.34346 training's l2: 5.49179 valid_1's rmse: 2.22034 valid_1's l2: 4.9299
[80] training's rmse: 2.29483 training's l2: 5.26627 valid_1's rmse: 2.17765 valid_1's l2: 4.74217
[100] training's rmse: 2.26635 training's l2: 5.13632 valid_1's rmse: 2.15028 valid_1's l2: 4.6237
[120] training's rmse: 2.23935 training's l2: 5.01469 valid_1's rmse: 2.12509 valid_1's l2: 4.51603
[140] training's rmse: 2.20977 training's l2: 4.8831 valid_1's rmse: 2.09643 valid_1's l2: 4.39503
[160] training's rmse: 2.19998 training's l2: 4.8399 valid_1's rmse: 2.0845 valid_1's l2: 4.34514
[180] training's rmse: 2.18051 training's l2: 4.75464 valid_1's rmse: 2.07392 valid_1's l2: 4.30116
[200] training's rmse: 2.17212 training's l2: 4.71809 valid_1's rmse: 2.06149 valid_1's l2: 4.24976
[220] training's rmse: 2.16078 training's l2: 4.66898 valid_1's rmse: 2.0583 valid_1's l2: 4.23662
[240] training's rmse: 2.14969 training's l2: 4.62118 valid_1's rmse: 2.0556 valid_1's l2: 4.22548
[260] training's rmse: 2.14074 training's l2: 4.58279 valid_1's rmse: 2.04951 valid_1's l2: 4.20051
[280] training's rmse: 2.13147 training's l2: 4.54316 valid_1's rmse: 2.04809 valid_1's l2: 4.19467
Early stopping, best iteration is:
[271] training's rmse: 2.13553 training's l2: 4.56051 valid_1's rmse: 2.04775 valid_1's l2: 4.19327
**\*** 商店预测: TX_2 **\***
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.155092 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 1362
[LightGBM] [Info] Number of data points in the train set: 5832737, number of used features: 20
[LightGBM] [Info] Start training from score 1.236878
Training until validation scores don't improve for 20 rounds
[20] training's rmse: 3.08819 training's l2: 9.53694 valid_1's rmse: 2.59737 valid_1's l2: 6.74635
[40] training's rmse: 2.93355 training's l2: 8.60572 valid_1's rmse: 2.51413 valid_1's l2: 6.32085
[60] training's rmse: 2.86407 training's l2: 8.20291 valid_1's rmse: 2.46019 valid_1's l2: 6.05256
[80] training's rmse: 2.79356 training's l2: 7.804 valid_1's rmse: 2.39947 valid_1's l2: 5.75744
[100] training's rmse: 2.74507 training's l2: 7.53539 valid_1's rmse: 2.33612 valid_1's l2: 5.45747
[120] training's rmse: 2.69753 training's l2: 7.27668 valid_1's rmse: 2.30495 valid_1's l2: 5.3128
[140] training's rmse: 2.67724 training's l2: 7.16762 valid_1's rmse: 2.2867 valid_1's l2: 5.22902
[160] training's rmse: 2.6668 training's l2: 7.11181 valid_1's rmse: 2.27585 valid_1's l2: 5.17949
[180] training's rmse: 2.64647 training's l2: 7.00378 valid_1's rmse: 2.26863 valid_1's l2: 5.14666
[200] training's rmse: 2.63547 training's l2: 6.94573 valid_1's rmse: 2.26284 valid_1's l2: 5.12046
[220] training's rmse: 2.62276 training's l2: 6.87885 valid_1's rmse: 2.25743 valid_1's l2: 5.09598
[240] training's rmse: 2.60309 training's l2: 6.77607 valid_1's rmse: 2.25208 valid_1's l2: 5.07185
[260] training's rmse: 2.5874 training's l2: 6.69464 valid_1's rmse: 2.25037 valid_1's l2: 5.06418
[280] training's rmse: 2.57257 training's l2: 6.6181 valid_1's rmse: 2.24126 valid_1's l2: 5.02326
[300] training's rmse: 2.56371 training's l2: 6.57262 valid_1's rmse: 2.23529 valid_1's l2: 4.99653
Early stopping, best iteration is:
[299] training's rmse: 2.56398 training's l2: 6.57397 valid_1's rmse: 2.23242 valid_1's l2: 4.9837
**\*** 商店预测: TX_3 **\***
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.102590 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 1361
[LightGBM] [Info] Number of data points in the train set: 5832737, number of used features: 20
[LightGBM] [Info] Start training from score 1.043992
Training until validation scores don't improve for 20 rounds
[20] training's rmse: 2.67913 training's l2: 7.17773 valid_1's rmse: 2.70167 valid_1's l2: 7.299
[40] training's rmse: 2.55842 training's l2: 6.5455 valid_1's rmse: 2.56179 valid_1's l2: 6.56277
[60] training's rmse: 2.49146 training's l2: 6.20739 valid_1's rmse: 2.51526 valid_1's l2: 6.32651
[80] training's rmse: 2.43299 training's l2: 5.91944 valid_1's rmse: 2.47552 valid_1's l2: 6.1282
[100] training's rmse: 2.39987 training's l2: 5.75939 valid_1's rmse: 2.4561 valid_1's l2: 6.03242
[120] training's rmse: 2.37242 training's l2: 5.62839 valid_1's rmse: 2.4159 valid_1's l2: 5.83657
[140] training's rmse: 2.34098 training's l2: 5.4802 valid_1's rmse: 2.398 valid_1's l2: 5.75041
[160] training's rmse: 2.31876 training's l2: 5.37666 valid_1's rmse: 2.38408 valid_1's l2: 5.68386
[180] training's rmse: 2.30461 training's l2: 5.31123 valid_1's rmse: 2.37582 valid_1's l2: 5.6445
[200] training's rmse: 2.29285 training's l2: 5.25714 valid_1's rmse: 2.36394 valid_1's l2: 5.58821
[220] training's rmse: 2.28143 training's l2: 5.20494 valid_1's rmse: 2.35974 valid_1's l2: 5.56836
[240] training's rmse: 2.27143 training's l2: 5.15941 valid_1's rmse: 2.35431 valid_1's l2: 5.54276
[260] training's rmse: 2.25815 training's l2: 5.09925 valid_1's rmse: 2.34868 valid_1's l2: 5.5163
[280] training's rmse: 2.25218 training's l2: 5.07231 valid_1's rmse: 2.34447 valid_1's l2: 5.49652
[300] training's rmse: 2.24322 training's l2: 5.03206 valid_1's rmse: 2.33699 valid_1's l2: 5.4615
[320] training's rmse: 2.23522 training's l2: 4.99621 valid_1's rmse: 2.33296 valid_1's l2: 5.44271
[340] training's rmse: 2.22802 training's l2: 4.96406 valid_1's rmse: 2.3299 valid_1's l2: 5.42841
[360] training's rmse: 2.22255 training's l2: 4.93972 valid_1's rmse: 2.33075 valid_1's l2: 5.43238
Early stopping, best iteration is:
[344] training's rmse: 2.22633 training's l2: 4.95654 valid_1's rmse: 2.32904 valid_1's l2: 5.42441
**\*** 商店预测: WI_1 **\***
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.145938 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 1357
[LightGBM] [Info] Number of data points in the train set: 5832737, number of used features: 20
[LightGBM] [Info] Start training from score 0.882787
Training until validation scores don't improve for 20 rounds
[20] training's rmse: 1.86792 training's l2: 3.48912 valid_1's rmse: 2.07401 valid_1's l2: 4.30151
[40] training's rmse: 1.80924 training's l2: 3.27335 valid_1's rmse: 2.01604 valid_1's l2: 4.06441
[60] training's rmse: 1.77502 training's l2: 3.15069 valid_1's rmse: 1.98087 valid_1's l2: 3.92383
[80] training's rmse: 1.73995 training's l2: 3.02742 valid_1's rmse: 1.95197 valid_1's l2: 3.81017
[100] training's rmse: 1.72376 training's l2: 2.97135 valid_1's rmse: 1.93543 valid_1's l2: 3.74587
[120] training's rmse: 1.70458 training's l2: 2.90558 valid_1's rmse: 1.92147 valid_1's l2: 3.69204
[140] training's rmse: 1.6924 training's l2: 2.86421 valid_1's rmse: 1.9138 valid_1's l2: 3.66262
[160] training's rmse: 1.68335 training's l2: 2.83368 valid_1's rmse: 1.89742 valid_1's l2: 3.60022
[180] training's rmse: 1.6732 training's l2: 2.79961 valid_1's rmse: 1.88665 valid_1's l2: 3.55944
[200] training's rmse: 1.66149 training's l2: 2.76054 valid_1's rmse: 1.8804 valid_1's l2: 3.5359
[220] training's rmse: 1.65349 training's l2: 2.73405 valid_1's rmse: 1.87548 valid_1's l2: 3.51744
[240] training's rmse: 1.64788 training's l2: 2.7155 valid_1's rmse: 1.8709 valid_1's l2: 3.50026
[260] training's rmse: 1.6378 training's l2: 2.68238 valid_1's rmse: 1.86663 valid_1's l2: 3.48429
[280] training's rmse: 1.63174 training's l2: 2.66258 valid_1's rmse: 1.86221 valid_1's l2: 3.46781
[300] training's rmse: 1.62809 training's l2: 2.65068 valid_1's rmse: 1.8607 valid_1's l2: 3.46222
[320] training's rmse: 1.62393 training's l2: 2.63716 valid_1's rmse: 1.85883 valid_1's l2: 3.45526
[340] training's rmse: 1.6209 training's l2: 2.62731 valid_1's rmse: 1.85764 valid_1's l2: 3.45083
[360] training's rmse: 1.61808 training's l2: 2.61818 valid_1's rmse: 1.85537 valid_1's l2: 3.44242
[380] training's rmse: 1.6152 training's l2: 2.60887 valid_1's rmse: 1.85231 valid_1's l2: 3.43105
[400] training's rmse: 1.61274 training's l2: 2.60094 valid_1's rmse: 1.85144 valid_1's l2: 3.42784
[420] training's rmse: 1.61064 training's l2: 2.59417 valid_1's rmse: 1.85094 valid_1's l2: 3.42598
[440] training's rmse: 1.60756 training's l2: 2.58426 valid_1's rmse: 1.85055 valid_1's l2: 3.42455
Early stopping, best iteration is:
[428] training's rmse: 1.60977 training's l2: 2.59135 valid_1's rmse: 1.85042 valid_1's l2: 3.42406
**\*** 商店预测: WI_2 **\***
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.094890 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 1358
[LightGBM] [Info] Number of data points in the train set: 5832737, number of used features: 20
[LightGBM] [Info] Start training from score 1.121945
Training until validation scores don't improve for 20 rounds
[20] training's rmse: 3.03694 training's l2: 9.22301 valid_1's rmse: 3.93361 valid_1's l2: 15.4733
[40] training's rmse: 2.9182 training's l2: 8.51589 valid_1's rmse: 3.78644 valid_1's l2: 14.3372
[60] training's rmse: 2.8478 training's l2: 8.10996 valid_1's rmse: 3.68479 valid_1's l2: 13.5777
[80] training's rmse: 2.79481 training's l2: 7.81098 valid_1's rmse: 3.58929 valid_1's l2: 12.883
[100] training's rmse: 2.7618 training's l2: 7.62757 valid_1's rmse: 3.52926 valid_1's l2: 12.4557
[120] training's rmse: 2.7385 training's l2: 7.49937 valid_1's rmse: 3.51672 valid_1's l2: 12.3674
[140] training's rmse: 2.7131 training's l2: 7.36093 valid_1's rmse: 3.4851 valid_1's l2: 12.1459
[160] training's rmse: 2.69602 training's l2: 7.26852 valid_1's rmse: 3.47113 valid_1's l2: 12.0487
[180] training's rmse: 2.67764 training's l2: 7.16978 valid_1's rmse: 3.45053 valid_1's l2: 11.9061
[200] training's rmse: 2.66142 training's l2: 7.08315 valid_1's rmse: 3.43415 valid_1's l2: 11.7934
[220] training's rmse: 2.64666 training's l2: 7.00478 valid_1's rmse: 3.42436 valid_1's l2: 11.7263
[240] training's rmse: 2.63366 training's l2: 6.93614 valid_1's rmse: 3.41016 valid_1's l2: 11.6292
[260] training's rmse: 2.6236 training's l2: 6.88328 valid_1's rmse: 3.40458 valid_1's l2: 11.5912
[280] training's rmse: 2.61225 training's l2: 6.82387 valid_1's rmse: 3.40012 valid_1's l2: 11.5608
[300] training's rmse: 2.60394 training's l2: 6.78053 valid_1's rmse: 3.39226 valid_1's l2: 11.5074
[320] training's rmse: 2.59456 training's l2: 6.73173 valid_1's rmse: 3.38642 valid_1's l2: 11.4678
[340] training's rmse: 2.58621 training's l2: 6.68846 valid_1's rmse: 3.37991 valid_1's l2: 11.4238
[360] training's rmse: 2.58159 training's l2: 6.6646 valid_1's rmse: 3.37706 valid_1's l2: 11.4046
[380] training's rmse: 2.57549 training's l2: 6.63315 valid_1's rmse: 3.36664 valid_1's l2: 11.3343
[400] training's rmse: 2.5691 training's l2: 6.60026 valid_1's rmse: 3.36251 valid_1's l2: 11.3065
[420] training's rmse: 2.56259 training's l2: 6.56688 valid_1's rmse: 3.35767 valid_1's l2: 11.2739
[440] training's rmse: 2.557 training's l2: 6.53827 valid_1's rmse: 3.35531 valid_1's l2: 11.2581
[460] training's rmse: 2.55061 training's l2: 6.50563 valid_1's rmse: 3.35751 valid_1's l2: 11.2729
Early stopping, best iteration is:
[448] training's rmse: 2.55554 training's l2: 6.53077 valid_1's rmse: 3.35281 valid_1's l2: 11.2414
**\*** 商店预测: WI_3 **\***
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.110176 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 1358
[LightGBM] [Info] Number of data points in the train set: 5832737, number of used features: 20
[LightGBM] [Info] Start training from score 1.102018
Training until validation scores don't improve for 20 rounds
[20] training's rmse: 2.97783 training's l2: 8.8675 valid_1's rmse: 2.89688 valid_1's l2: 8.39191
[40] training's rmse: 2.82641 training's l2: 7.98859 valid_1's rmse: 2.72241 valid_1's l2: 7.41149
[60] training's rmse: 2.74667 training's l2: 7.54421 valid_1's rmse: 2.65431 valid_1's l2: 7.04535
[80] training's rmse: 2.69562 training's l2: 7.26634 valid_1's rmse: 2.61797 valid_1's l2: 6.85378
[100] training's rmse: 2.67057 training's l2: 7.13195 valid_1's rmse: 2.59677 valid_1's l2: 6.7432
[120] training's rmse: 2.63212 training's l2: 6.92807 valid_1's rmse: 2.56413 valid_1's l2: 6.57477
[140] training's rmse: 2.60835 training's l2: 6.80348 valid_1's rmse: 2.53236 valid_1's l2: 6.41285
[160] training's rmse: 2.58137 training's l2: 6.66347 valid_1's rmse: 2.50533 valid_1's l2: 6.27668
[180] training's rmse: 2.57078 training's l2: 6.6089 valid_1's rmse: 2.49547 valid_1's l2: 6.22735
[200] training's rmse: 2.55541 training's l2: 6.5301 valid_1's rmse: 2.4852 valid_1's l2: 6.17623
[220] training's rmse: 2.54417 training's l2: 6.47281 valid_1's rmse: 2.48101 valid_1's l2: 6.15541
[240] training's rmse: 2.52961 training's l2: 6.39891 valid_1's rmse: 2.46351 valid_1's l2: 6.06888
[260] training's rmse: 2.51821 training's l2: 6.34138 valid_1's rmse: 2.45271 valid_1's l2: 6.01577
[280] training's rmse: 2.5091 training's l2: 6.2956 valid_1's rmse: 2.44894 valid_1's l2: 5.99729
[300] training's rmse: 2.49981 training's l2: 6.24906 valid_1's rmse: 2.4397 valid_1's l2: 5.95212
[320] training's rmse: 2.48997 training's l2: 6.19993 valid_1's rmse: 2.43379 valid_1's l2: 5.92334
[340] training's rmse: 2.48433 training's l2: 6.17192 valid_1's rmse: 2.43239 valid_1's l2: 5.91652
[360] training's rmse: 2.47665 training's l2: 6.13382 valid_1's rmse: 2.42876 valid_1's l2: 5.8989
[380] training's rmse: 2.4723 training's l2: 6.11226 valid_1's rmse: 2.42443 valid_1's l2: 5.87785
[400] training's rmse: 2.46662 training's l2: 6.08421 valid_1's rmse: 2.42119 valid_1's l2: 5.86215
[420] training's rmse: 2.45896 training's l2: 6.0465 valid_1's rmse: 2.4151 valid_1's l2: 5.83269
[440] training's rmse: 2.45114 training's l2: 6.00808 valid_1's rmse: 2.40898 valid_1's l2: 5.80318
[460] training's rmse: 2.44734 training's l2: 5.98948 valid_1's rmse: 2.40707 valid_1's l2: 5.79397
[480] training's rmse: 2.43899 training's l2: 5.94866 valid_1's rmse: 2.40305 valid_1's l2: 5.77464
[500] training's rmse: 2.43483 training's l2: 5.92838 valid_1's rmse: 2.40297 valid_1's l2: 5.77425
Early stopping, best iteration is:
[486] training's rmse: 2.43842 training's l2: 5.94589 valid_1's rmse: 2.40268 valid_1's l2: 5.77287

=== 特征重要性分析 ===
分析特征重要性...

=== 准备提交文件 ===
准备提交文件...
提交文件已保存，包含 60980 行数据
提交文件已生成：submission.csv，包含 60980 行数据

分析完成！
最终数据框形状: (60034810, 21)

"""
