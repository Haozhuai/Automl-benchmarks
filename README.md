# AutoML Benchmarks of Tabular Dataset

    Empirical Comparison among popular autoML toolkit include google autoML, autogluon, tpot, h2o, etc.
    
## Requirement
    google cloud account # 300$ credit for free
    autogluon  # pip install autogluon (may cause issues on Windows)
    pytorch-tabnet # pip install pytorch-tabnet
    tpot  # pip install tpot
    h2o # python package, java should be installed before h2o
    mxnet # pip install mxnet
    
## step

    git clone https://github.com/Haozhuai/Automl-benchmarks.git
    cd Automl-benchmarks  # should decompress dataset.7z ！！
    python tabnet.py  # tabnet benchmarks 
    python wide_deep.py # wide&deep neural benchmarks(from autogluon's tabular neural network)
    python tpot_benchmarks.py # tpot benchmarks
    sh h2o_run.sh  # h2o benchmarks
    python google_split.py  # google dataset split, should pass the folder to google cloud and train each dataset by hand
    
## result-table(Binary Classification: metric here is ROC-AUC)
    autogluon is open-source automl from amazon
    google is google cloud automl products(26 datasets cost me 4000+HKD)
    tpot is a popular automl toolkit
    h2o is from h2o.ai, a famous AI company
    tabnet is introduced by google cloud team, try to find a neural network architecture for tabular data
    wide&deep is successful in recommendation system area
    default_lgb: default configuration of LightGBM
    default_cat: default configuration of CatBoost
    

|dataset|autogluon|google|tpot|h2o|tabnet|wide_deep|default_lgb|default_cat|
|---|---|---|---|---|---|---|---|---|
|Amazon_employee_access|0.8550651417683003|0.8385228833425236|0.8510637793216722|0.4986252575388828|0.7183694434883128|0.7582595551716692|0.8418818469100458|0.8307555409995829|
|BayesianNetworkGenerator_vote|0.9957191438428682|0.9958579988155696|0.9958290266867248|0.995856194620396|0.9957451500213108|0.995758541704307|0.9958508324923372|0.9959734277589356|
|Click_prediction_small|0.7400046815287358|0.7375588310424078|0.7191765421007239|0.734635|0.687861046782622|0.719329478972208|0.7360879220117181|0.7357084039145705|
|CreditCardSubset|0.8611962652821972|0.8782113548819293|0.9487523027968516|0.8536593535421202|0.9714620666555016|0.8664545302294423|0.886417685479819|0.8646792832021436|
|MagicTelescope|0.9479639163590312|0.9446325887203256|0.9371122443480372|0.9425244536660292|0.940876020586888|0.9410980571798118|0.9406977715502142|0.944592169437633|
|New_KDDCup09_appetency|0.8547957968696735|0.8450918229631001|0.7715330383955507|0.831000040419848|0.8179291768002276|0.7987150047372571|0.8217182197675427|0.8545200702339343|
|New_KDDCup09_churn|0.7532905960956402|0.7471244234383453|0.725345915366271|0.7377763163771099|0.6638858846890451|0.6677833497305623|0.720765225199202|0.752528729538178|
|New_KDDCup09_upselling|0.875257944766746|0.8704324450183096|0.8680855609052379|0.8565444127660753|0.7990857090753372|0.8075425419150415|0.850883696241527|0.8752551700781068|
|New_aps_failure|0.9922370442903706|0.9897445818755192|0.9913988791672376|0.9896946027778404|0.9867248578035044|0.990153033611973|0.9891621248206824|0.9914340596817643|
|New_higgs|0.8101991827855457|0.8163452199586023|0.7892442136865327|0.8073084977654137|0.808996716927241|0.7956089136071967|0.8030723216805379|0.8029898928274635|
|New_kick|0.7698127222211544|0.7871217733113809|0.7804580829303855|0.7864488795475684|0.748018689877621|0.7617994075765641|0.7674757545717655|0.7816949574312894|
|New_test_dataset|0.9363993333026804|0.9408183364752404|0.9130325072417124|0.9342607016414548|0.915299094211228|0.9066971355004828|0.9203593728447286|0.9229216975493126|
|Run_or_walk_information|0.9993776756663424|0.9992782179245578|0.9992340575807166|0.999293898867376|0.9990535879939588|0.9991084939459612|0.9993982063017456|0.9992991201852808|
|adult-census|0.9349402962856978|0.932394290465276|0.9308126715699088|0.9334182068405088|0.9212610485747488|0.9208361364634838|0.9328245050210996|0.9332264016567932|
|ailerons|0.961303332277576|0.9574008013485452|0.957792618813592|0.9612784224100336|0.9467366514913612|0.957756397358746|0.9601566403433359|0.9601679520269092|
|bank-marketing|0.9441631037360568|0.9426165956920478|0.9417120053511512|0.9403374952170194|0.935843426849582|0.934596024442774|0.9426432693748924|0.9427422829273858|
|dataset_3_kr-vs-kp|0.9998869257118952|0.9999564190396499|0.9999651352317199|1.0|0.9950840676725152|0.9999433447515448|0.99992155427137|0.9999825676158599|
|eeg-eye-state|0.9984728894225486|0.9978557313301212|0.992917688545226|0.9913416620814316|0.7221914093547599|0.9977844758791292|0.9882494557264464|0.9890272610145217|
|elevators|0.9542078260301424|0.9559417935196092|0.9475530992225956|0.9557752432681692|0.6762234787135749|0.952160234182494|0.9427373043671856|0.9455928291996958|
|fried|0.9872655799548996|0.9879746841535368|0.9855681075822404||0.9876657201932534|0.9866095607861392|0.9854867421706314|0.9871622065337041|
|kdd_ipums_la_97-small|0.995887162015672|0.994274979093753|0.9959610292068016|0.9956203348096756|0.9912880958280422|0.9963704432124384|0.9951480084863876|0.9958584337349398|
|letter|0.999859415122068|0.9999558550450564|0.9998604449811458|0.9997116337620616|0.9998996058282732|0.9998924856742499|0.9998647170735596|0.9998120279337884|
|nomao|0.9962638632424201|0.9952580500491179|0.993889077887399|0.99570175977045|0.9926289817033804|0.9922939645896814|0.995673154039888|0.995671091126626|
|pendigits|0.9998930936841742|0.9999625085712311|0.9998401681194596|0.9999871739848948|0.9999763212028828|0.9999871739848948|0.9999644818043242|0.999952642405766|
|skin-segmentation|0.9998901780911968|0.9999908058114022|0.9999928430897028||0.9999287470070232|0.9999612433395848|0.9999873241016614|0.9999867076309223|
|sylva_prior|0.9990262389875504|0.999050179111732|0.9992496043958898|0.9989500026899224|0.9992616626688854|0.9991791098768388|0.9986921411597091|0.9989407270953105|

    
    