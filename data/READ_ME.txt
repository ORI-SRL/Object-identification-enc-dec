The current commit uses the validation_data_523 as the 5:2:3 split that we discussed on 4/1 (or 1/4 depending on your view)
Run the main file with the TUNING flag True and the USE_PREVIOUS flag false. 
All of the functions used at the moment are within the online_validation.py file
The PREP_TUNE needs to be TRUE to enable the batches to be organised in the organise_tuning_data function
- This is a very slow function at the moment because of the size of the arrays once looped over each epoch

with the USE_PREVIOUS flag low, the model will train that combined data from scratch, which usually takes about 140 epochs. It is processed in the tune_RNN_network function
Once the model has trained it gets passed to the test_tuned_model to return labels for confusion matrix processing.

Hopefully the comments help along the way but let me know if anything is unclear.

The network is the IterativeRNN2 within networks.py and the dropout has been reduced at the moment to 0.05

