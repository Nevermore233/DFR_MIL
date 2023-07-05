# DFR_MIL
DFR-MIL: an effectiveness-driven and statistically interpretable framework for predicting drug failure risk with the ability to determine optimal positive threshold

Package Dependency
pandas: 1.4.2
numpy: 1.21.5
scipy 1.7.3
scikit-learn 1.0.2

The results of the paper(main.py)：

MIL_600
##### huber loss #####
test_mse:0.012375668651331453:HL_value:3.2043839186779497,p_val:0.9208841097252112

##### mse loss #####
test_set
test_mse:0.012712797231483011:HL_value:3.3026633810305035,p_val:0.9139539800182652

##### mae loss #####
test_mse:0.011530147010584602:HL_value:2.5044893582901415,p_val:0.9615212706229984

##### logcosh loss #####
test_mse:0.012810445133380517:HL_value:3.2619799443407755,p_val:0.9168575982627889

##### HL loss #####
test_mse:0.007901406177062672:HL_value:0.6838749245365412,p_val:0.9995660250678401

MIL_1000
##### huber loss #####
test_mse:0.009579452707887074:HL_value:2.955468432137594,p_val:0.9371209527780583

##### mse loss #####
test_mse:0.010270532649810076:HL_value:3.3949902550543474,p_val:0.9071849043594742

##### mae loss #####
test_mse:0.009437195204388949:HL_value:2.938489434677021,p_val:0.9381581065648636

##### logcosh loss #####
test_mse:0.01029254884405355:HL_value:3.405847765587141,p_val:0.90637270155496

##### HL loss #####
test_mse:0.008824820592601616:HL_value:2.3470520313693157,p_val:0.9684819412680448

Step 1: Data Processing
python data_prepare.py

Step 2: run the code
python main.py

Step 3: partial result figures
python Figures.py

Citation
To be added...
