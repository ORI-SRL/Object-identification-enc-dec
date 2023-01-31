from utils.data_handlers import *
from utils.networks import *
from utils.experiment_runs import *
plt.rcParams.update({'font.size': 15})


DATA_PATH = os.path.abspath(os.getcwd())
DATA_FOLDER = "./data/validation_data_523/"
MODEL_SAVE_FOLDER = './saved_model_states/iterative/shifted/'
FILE_PREFIX = "shifted_"
n_grasps = [10]  # , 7, 5, 3, 1]
loss_comparison_dict = {}
sil_comparison_dict = {}
ml_dict = {}
train_ratio, valid_ratio = .6, .2  # test will be the remaining .2

SEED = 1234
seed_experiment(SEED)
classes = ['apple', 'bottle', 'cards', 'cube', 'cup', 'cylinder', 'sponge'] # Define the object classes
batch_size = 32
n_epochs = 1000
noise_level = .0
criterion = nn.CrossEntropyLoss()

TRAIN_MODEL = True
TEST_MODEL = False
USE_PREVIOUS = True
JOINT_DATA = False
COMPARE_LOSSES = True
ITERATIVE = True
RNN = True
ONLINE_VALIDATION = False
TUNING = True


def get_model(Model, use_previous=False, save_folder=''):
    model = Model()
    model_name = model.__class__.__name__.split('_')[0]
    exist = False

    model_state = None
    if use_previous:
        '''either use the previous model and update, or train from scratch'''
        model_state = f'{save_folder}{model_name}_dropout_model_state.pt'
        if exists(model_state):
            model.load_state_dict(torch.load(model_state))
            print("Model loaded!")
            exist = True
        else:
            print("Could not find model to load!")

    return model, model_state, model_name, exist


'''This is where the model can either be tuned and updated, or learnt from scratch with the combined data'''
if TUNING:
    PREP_TUNE = True

    print("Creating 'old' dataset splits...")
    old_data = GraspDataset(x_filename="data/raw_data/base_unshuffled_original_data.npy",
                            y_filename="data/raw_data/base_unshuffled_original_labels.npy")
    old_train_data, old_valid_data, old_test_data = old_data.get_splits(train_ratio=train_ratio,
                                                                        valid_ratio=valid_ratio)

    print("Creating 'new' dataset splits...")
    new_data = GraspDataset(x_filename="data/raw_data/base_unshuffled_tuning_data.npy",
                            y_filename="data/raw_data/base_unshuffled_tuning_labels.npy",
                            normalize=True)
    new_train_data, new_valid_data, new_test_data = new_data.get_splits(train_ratio=train_ratio,
                                                                        valid_ratio=valid_ratio)
    print("Datasets ready!")

    model, model_state, model_name, exist = get_model(IterativeRNN4,
                                                      use_previous=USE_PREVIOUS,
                                                      save_folder=MODEL_SAVE_FOLDER)

    # ---------------  OPTIMIZE EMBEDDINGS BY BACKPROP THORUGH INPUTS -----------------------
    # if exist:
        # grasp_pred_labels = test_tuned_model(model, n_epochs, batch_size, criterion,
        #                                      old_data=(old_train_data, old_valid_data, old_test_data),
        #                                      new_data=(new_train_data, new_valid_data, new_test_data),
        #                                      oldnew=JOINT_DATA,
        #                                      noise_level=noise_level,
        #                                      save_folder=MODEL_SAVE_FOLDER,
        #                                      show=True,
        #                                      save=True)
    #     plot_embed(model, old_train_data, batch_size, device=get_device(), show=True, save=False)
    #     plot_embed_optimize(model, model_state, data=old_data, device=get_device(), show=True, save=False)


    # ---------------  CHECK INFORMATIONA LOADINGS -----------------------

    attention_analysis(model, old_test_data, batch_size,
                       device=get_device(),
                       save_folder='./figures/',
                       show=True,
                       save=True)


    # ---------------  OPTIMIZE EMBEDDINGS BY BACKPROP THORUGH EMBED LAYER -----------------------
    # model, model_state, model_name, exist = get_model(IterativeRNN4_embed,
    #                                                   use_previous=USE_PREVIOUS,
    #                                                   save_folder=MODEL_SAVE_FOLDER)
    # if exist:
    #     plot_embed_optimize_direct(model, model_state, data=old_data, device=get_device(), show=True, save=True)
