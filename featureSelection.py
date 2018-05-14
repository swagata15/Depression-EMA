

# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class','prege', 'plase', 'prese', 'skine', 'teste', 'masse', 'pedie', 'agee', 'classe','prega', 'plasa', 'presa', 'skina', 'testa', 'massa', 'pedia', 'agea', 'classa','pregs', 'plass', 'press', 'skins', 'tests', 'masss', 'pedis', 'ages', 'classs','preges', 'plases', 'preses', 'skines', 'testes', 'masses', 'pedies', 'agees', 'classes','pregas', 'plasas', 'presas', 'skinas', 'testas', 'massas', 'pedias', 'ageas', 'classas','pregd', 'plasd', 'presd', 'skind', 'testd', 'massd', 'pedid', 'aged', 'classd','pregf', 'plasf', 'presf', 'skinf', 'testf', 'massf', 'pedif']
#url = "/Users/swagataashwani/Desktop/DepressionEMA/Week7/Evening/eveningConcatenated.csv"
#names =['Activityclass	','Activityresearch','	Activitystudying','	Activityinclass','	Activitychores','	Activityselfcare','	Activityeat','	Activitydrink	','Activityalcohol	','Activitysmoke','	Activityjob','	Activityexercise','	Activitysocial','	Activitysleep','	Activityhobby','	Activitycommute','	Activityother','	ActivityAcademiccat','	ActivityPersonalcat	','Anxious','	Depressed	','Frustrated	','Happy	','Overwhelmed','	Lonely','	Connected	','NumberSocialInt','	FacetoFaceInt','	LikedtohavesocialInt','	Stressdemands','	Demandsacadeload','	Demandsacadeperfor	','Demandsinstructo','	Demandsmajor','	Demandsfamily	','Demandsconflict	','Demandsfinancial','	Demandsliving','	Demandshealth','	Demandssafety','	Demandsother','	DemandsNOSTRESS','	DemandsAcademic','	DemandsPersonal','	Demandstotal','	CopingStrategy','	Successcoping','	SuccesscopingR','	Control','	controlrecoded','	Emotions	','EmotionsR','	Resources	Resourcesrecoded','	Belonging	Discrimination','	DiscriReason','	Productive','	Hrsacademic','	hrsnumacademic','	Hrsschedule','	hrsnumschedule','	ManyThingsToDo','	EmotionallyDrained','	CareSelf	','ProfessorCares','	TalkStress','	LittleSleep','	SocialSupport	','Exercise	','Concentrating','	PainInterfere','	Homesick']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)


#url = "/Users/swagataashwani/Desktop/DepressionEMA/Week7/Afternoon2/Afternoon2Concatenated.csv"
#names = ['Activity_class'	'Activity_research'	'Activity_studying'	'Activity_inclass'	 'Activity_chores'	'Activity_selfcare'	'Activity_eat'	'Activity_drink'	'Activity_alcohol'	'Activity_smoke'	'Activity_job'	'Activity_exercise'	'Activity_social'	'Activity_sleep'	'Activity_hobby'	'Activity_commute'	'Activity_other'	'Activity_Academic_cat'	'Activity_Personal_cat'	'Anxious'	 'Depressed'	'Frustrated'	'Happy'	'Overwhelmed' 	'Lonely' 	'Connected'	'Number_Social_Int'	'Face_to_Face_Int'	'Liked_to_have_social_Int'	 'Stress_demands' 	'Demands_acade_load'	'Demands_acade_perfor' 	'Demands_instructor'	'Demands_major'	'Demands_family'	'Demands_conflict' 	'Demands_financial'	'Demands_living'	'Demands_health'	'Demands_safety'	'Demands_Academic'	'Demands_Personal'	'Demands_total'	'Success_coping'	'Control'	'Resources'	'ThinkToDo'	'LookingForward'	'TimeWisely' 	 'WorriedGrades' 'Depressedornot']
