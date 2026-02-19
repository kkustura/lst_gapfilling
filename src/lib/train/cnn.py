#### Functions for the pixelwise CNN training ####

import sys
import os
import joblib
import psutil
import yaml
import tensorflow as tf
from datetime import datetime
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.layers import Dense, BatchNormalization, Dense, Dropout, Flatten, Conv2D, Input, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.base import BaseSetup
from lib.data.image import ImageProcessor
from lib.train.arrays import NetPreprocessor
from lib.utils import locate_single_file

class CNNModel(BaseSetup):
    def __init__(self, cfg, cnn_preparator=None, model_tag=None):
        """
        Initialize the CNNModel with cnn_preparator and configuration path.
        If cnn_preparator is not provided, it raises a ValueError.
        """
        super().__init__(cfg)
        self.config = cfg     
        
        if cnn_preparator is not None:
            self._init_new_model(cnn_preparator, model_tag=model_tag)
        else:
            self._load_existing_model(model_tag=model_tag)
            
        self.n_patches = self.metadata['n_patches']
        self.N_num = self.metadata['N_num']
        self.N_cat = self.metadata['N_cat']
        self.target_nodata = self.metadata['target_nodata']
        self.patch_size = self.metadata['patch_size']
        self.target_dataset = self.metadata['target_dataset']
        self.nodata = self.metadata['nodata']  # dict of no-data values for each dataset
        self.nodata.update({self.target_dataset : self.metadata['target_nodata']})  
        self.numerical_datasets = self.metadata['input_datasets'][:self.N_num]  
        self.categorical_datasets = self.metadata['input_datasets'][self.N_num:]  
        
        self.max_epochs = self.max_epochs
        self.dropout_rate = self.dropout_rate
        self.validation_split = self.validation_split
        self.train_valid_split_shuffle = self.train_valid_split_shuffle
 
        # TBD!!!! hardcoded values 
        self.batch_size = 512 
        self.units_1 = 128
        self.units_2 = 512 
        self.activation  ='linear' 
        self.lr = 1e-4
        self.normalize = True
        self.n_train = int(self.n_patches*self.validation_split)
        self.steps_per_epoch = self.n_train // self.batch_size
        self.buffer_size = 1e4
        self.norm = ()
        
        self.kernel_size = (self.patch_size//2, self.patch_size//2)
        self.strides = (1, 1)
    
    
    def _init_new_model(self, cnn_preparator, model_tag=None):
        """
        Initialize a new CNN model with the given cnn_preparator.
        If model_tag is not provided, it will be set to the current date and time.
        """
        if not hasattr(cnn_preparator, 'metadata'):
            raise ValueError("'metadata' attribute is mising in cnn_preparator. Please run prepare_input_arrays() or load_input_arrays() before initializing CNNModel.")
        self.model_tag = datetime.now().strftime("%Y%m%dT%H%M%S") if model_tag is None else model_tag
        self.metadata = cnn_preparator.metadata
        self.logger.info(f"Initialized CNN model with the tag: {self.model_tag}")
    
    
    def _load_existing_model(self, model_tag=None):
        """Load a pre-trained model and its metadata."""                
        net_output_dir = self.config['net_output_directory']
        models = os.listdir(net_output_dir)
        models = sorted(models, key = lambda x: x, reverse=True)
        
        if not models:
            raise FileNotFoundError(f"No model files found in {net_output_dir}. Please train a model first.")
        if model_tag is None:
            model_tag = models[0]
            self.logger.warning(f"No model_tag to load is provided. Using the latest model in the directory: {model_tag}")
        elif model_tag not in models:
            raise ValueError(f"Model {model_tag} not found in {net_output_dir}. Available models: {models}")
        
        self.model_dir = os.path.join(net_output_dir, model_tag)
        self.ensure_dirs_exist(self.model_dir)
        path_to_model    = locate_single_file(self.model_dir, patterns_in=[f"{model_tag}_model.keras"])
        path_to_scalers  = locate_single_file(self.model_dir, patterns_in=[f"{model_tag}_scalers.pkl"])
        path_to_metadata = locate_single_file(self.model_dir, patterns_in=[f"{model_tag}_metadata.yml"])
        
        with open(path_to_scalers, 'rb') as file:
            self.scalers = pickle.load(file)
        
        with open(path_to_metadata, 'r') as file:
            mtd = yaml.safe_load(file)    
            
        input_datasets = NetPreprocessor(self.config_path).update_input_datasets()
        if set(input_datasets) != set(mtd['input_datasets']):
            raise ValueError(f"Input datasets in metadata ({mtd['input_datasets']}) do not match the current configuration ({input_datasets}).")    
        if mtd['patch_size'] != self.patch_size:
            raise ValueError(f"Metadata patch_size ({mtd['patch_size']}) does not match the current configuration ({self.patch_size}).")
        if mtd['target_nodata'] != self.target_nodata:
            raise ValueError(f"Metadata target_nodata ({mtd['target_nodata']}) does not match the current configuration ({self.target_nodata}).")

        self.model = load_model(path_to_model, compile=False)
        self.model_tag = model_tag
        self.metadata = mtd
        self.logger.info(f"Loaded model with the tag: {self.model_tag}")
    

        
    def normalize_numerical_arrays(self, X, Y, scalers={}):
        """
        Normalize Y and numerical features in X using MinMaxScaler.
        If X_scalers dictionary is provided, it will be used for normalization. 
        If X_scalers={}, new scalers are initialized and calculated.
        Returns a dictionary of scalers (unchanged scalers are returned if provided, 
        else new scalers are created).  
        """
        def _normalize_array(arr, nv, scaler=None):
            """
            Normalize a single array arr with a corresponding no-data value nv. Changes array in-place.
            If scaler is provided, it is used for normalization. If None, a new scaler is initialized and calculated.
            Output: scaler.
            """
            arr_shape = arr.shape                           # define array shape
            if arr_shape == (0,):
                return scaler
            arr_flat = arr.reshape(-1, 1)                   # flatten array
            arr_flat[np.where(arr_flat == nv)] = np.nan     # change no-data values by nan
            if scaler == None: 
                scaler = MinMaxScaler()                     # initialize scaler
                arr_flat = scaler.fit_transform(arr_flat)   # normalize array
            else: 
                arr_flat = scaler.transform(arr_flat)       # normalize array using a learned scaler
            arr_flat = ImageProcessor.set_nan_values(arr_flat, nv)         # put back no-data values
            arr[:] = arr_flat.reshape(arr_shape)            # transform into original shape
            return scaler
        
        scalers_out = {}
        
        X_num = X[:,:,:,:self.N_num]                            # select only numerical features from X
        num_patches, patch_size, _, num_features = X_num.shape  # define relevant values from X_num.shape
        X_num = X_num.reshape(-1, num_features)                 # flatten the array for each feature separately
        scalers_exist = True if scalers else False
        
        # normalize Y
        y_scaler = scalers[self.target_dataset] if scalers_exist else None
        scalers_out[self.target_dataset] = _normalize_array(Y, self.nodata[self.target_dataset], scaler=y_scaler)
            
        for i in range(num_features):
            key = self.numerical_datasets[i]    # define key
            arr = X_num[:, i]                   # read feature array
            nv = self.nodata[key]               # define no-data value for this feature
            arr[np.where(arr == nv)] = np.nan   # change no-data values by nan
            x_scaler = scalers[key] if scalers_exist else None  # get scaler for this feature if exists
            scalers_out[key] = _normalize_array(arr, nv, scaler=x_scaler)  # normalize array
            X_num[:, i] = arr.flatten()         # modify original array
        X[:,:,:,:self.N_num] = X_num.reshape(num_patches, patch_size, patch_size, num_features) 
        return scalers_out

    
    def _custom_optimizer(self, learning_rate=1e-4, decay_rate=1):
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=learning_rate,
            decay_steps=self.steps_per_epoch*1000,
            decay_rate=decay_rate,
            staircase=False)
        optimizer = Adam(learning_rate=lr_schedule)
        return optimizer
    

    def cnn_model(self, input_shape):
        # Numerical branch with BatchNormalization
        numerical_input = Input(shape=(input_shape[0], input_shape[1], self.N_num))
        if self.normalize:
            x_numerical = BatchNormalization()(numerical_input)
        x_numerical = Conv2D(self.units_1, kernel_size=self.kernel_size, strides=self.strides, activation='relu')(x_numerical)
        x_numerical = Conv2D(self.units_1, kernel_size=self.kernel_size, strides=self.strides, activation='relu')(x_numerical)
        if self.normalize:
            x_numerical = BatchNormalization()(x_numerical)
        if self.dropout_rate is not None and self.dropout_rate > 0:
            x_numerical = Dropout(self.dropout_rate)(x_numerical)
        x_numerical = Conv2D(self.units_2, kernel_size=self.kernel_size, strides=self.strides, activation='relu')(x_numerical)
        x_numerical = Conv2D(self.units_2, kernel_size=self.kernel_size, strides=self.strides, activation='relu')(x_numerical)
        if self.normalize:
            x_numerical = BatchNormalization()(x_numerical)
        if self.dropout_rate is not None and self.dropout_rate > 0:
            x_numerical = Dropout(self.dropout_rate)(x_numerical)
        x_numerical = Flatten()(x_numerical)
        
        # Categorical branch (no BatchNormalization for categorical features)
        if self.N_cat > 0:
            categorical_input = Input(shape=(input_shape[0], input_shape[1], self.N_cat))
            x_categorical = Conv2D(self.units_1, kernel_size=self.kernel_size, strides=self.strides, activation='relu')(categorical_input)
            x_categorical = Conv2D(self.units_1, kernel_size=self.kernel_size, strides=self.strides, activation='relu')(x_categorical)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                x_categorical = Dropout(self.dropout_rate)(x_categorical)
            x_categorical = Conv2D(self.units_2, kernel_size=self.kernel_size, strides=self.strides, activation='relu')(x_categorical)
            x_categorical = Conv2D(self.units_2, kernel_size=self.kernel_size, strides=self.strides, activation='relu')(x_categorical)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                x_categorical = Dropout(self.dropout_rate)(x_categorical)
            x_categorical = Flatten()(x_categorical)
            # Concatenate the outputs of the numerical and categorical branches
            merged = Concatenate()([x_numerical, x_categorical])
        else:
            # If no categorical features, just use the numerical features
            merged = x_numerical
            
        # Dense layers and output
        x = Dense(self.units_1, activation='relu')(merged)
        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)
        output = Dense(1, activation=self.activation)(x)

        # Create the model with appropriate inputs
        inputs = [numerical_input, categorical_input] if self.N_cat > 0 else numerical_input
        model = Model(inputs=inputs, outputs=output)

        # Configure optimizer
        optimizer = self._custom_optimizer(learning_rate=self.lr)

        model.compile(loss='mean_absolute_error', optimizer=optimizer)
        # model.compile(loss=ecostress_custom_loss, optimizer=optimizer, metrics=['loss', 'accuracy'])
        return model
    
    @staticmethod
    def _patch_middle_value(Y):
        # skip patching if no data is provided
        if Y.size == 0:
            return Y
        patch_size = Y.shape[1]
        if patch_size%2 == 0:
            raise ValueError("Patch size must be odd to obtain middle value.")
        if patch_size < 3:
            raise ValueError("Patch size must be at least 3 to obtain middle value.")
        middle_index = patch_size // 2
        Y_1d = np.zeros(Y.shape[0]).reshape(-1,1)
        for i in range(len(Y_1d)):
            Y_1d[i] = Y[i,middle_index,middle_index]
        return Y_1d 
    
    
    def save(self):   
        # save model
        self.model_path = os.path.join(self.model_dir, f"{self.model_tag}_model.keras")
        self.model.save(self.model_path)
        joblib.dump(self.history, os.path.join(self.model_dir, "history.gz"))
        self.logger.info(f"Model saved as: {self.model_path}")
        
        # generate metadata
        self.metadata.pop('type')
        if 'time_created' in self.metadata:
            self.metadata['time_data_generated'] = self.metadata.pop('time_created')
        try:
            self.metadata['time_model_generated'] = datetime.strptime(self.model_tag, "%Y%m%dT%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            self.metadata['time_model_generated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.metadata['model_path'] = self.model_path
        self.metadata['max_epochs'] = self.max_epochs
        self.metadata['dropout_rate'] = self.dropout_rate
        self.metadata['validation_split'] = self.validation_split
        self.metadata['batch_size'] = self.batch_size
        self.metadata_path = os.path.join(self.model_dir, f"{self.model_tag}_metadata.yml")
        with open(self.metadata_path, 'w') as file:
            yaml.dump(self.metadata, file, sort_keys=False, default_flow_style=False)
    
    
    def plot_history(self, log_scale=False):
        """Monitor the training and validation loss."""       
        plt.figure(figsize=(5, 3))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        if log_scale:
            plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, f"{self.model_tag}_loss_history.png"))
        plt.show()
        plt.close()
    
    
    def evaluate_model(self, X_test, Y_test):
        if self.model is None:
            raise ValueError("Model has not been trained yet.") 
        
        # split into numerical and categorical parts if not already done
        if self.N_cat > 0 and not isinstance(X_test, list):
            X_test = [X_test[:, :, :, :self.N_num], X_test[:, :, :, self.N_num:]]

        scores = self.model.evaluate(X_test, Y_test, verbose=1)
        self.logger.info(f"Scores: {scores}")
        
        Y_pred = self.model.predict(X_test)
        r2 = r2_score(Y_test, Y_pred)
        self.logger.info(f"R2 Score: {r2}")
        
        x_line = np.arange(0,1,0.1)
        plt.figure(figsize=(5,3))
        plt.plot(x_line, x_line, color='k', linestyle='--')
        plt.scatter(Y_test, Y_pred, alpha=0.05, label=f"R2 score={r2:.2f}")
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, f"{self.model_tag}_r2.png"))
        plt.show()
        plt.close()
        return scores, r2
    
    
    


    def train(self, X, Y):
        """Train the CNN model with with input data X (4D array) and target data Y (1D array)."""
        # define model directory
        self.model_dir = os.path.join(self.net_output_directory, self.model_tag)
        self.ensure_dirs_exist(self.model_dir)
        
        # split X and Y into training and validation sets
        X_train, X_valid, Y_train, Y_valid = train_test_split(
            X, Y,
            test_size=self.validation_split, 
            shuffle=self.train_valid_split_shuffle, 
            random_state=42
        )
        # normalize numerical features and generate scalers
        self.scalers = self.normalize_numerical_arrays(X_train, Y_train, scalers={})
        self.normalize_numerical_arrays(X_valid, Y_valid, scalers=self.scalers)
        
        # TBD!!!! transfer this to array prep (before saving!)
        Y_train = CNNModel._patch_middle_value(Y_train)
        Y_valid = CNNModel._patch_middle_value(Y_valid)
        
        # save scalers to a .pkl file
        with open(os.path.join(self.model_dir, f'{self.model_tag}_scalers.pkl'), 'wb') as f:
            pickle.dump(self.scalers, f)
        self.logger.info(f'Scalers saved to {self.model_dir}')
        
        # generate the model
        self.model = self.cnn_model(input_shape=X_train.shape[1:])
           
        # split X_train into numerical and categorical parts
        X_train = [X_train[:, :, :, :self.N_num], X_train[:, :, :, self.N_num:]] if self.N_cat > 0 else X_train
        X_valid = [X_valid[:, :, :, :self.N_num], X_valid[:, :, :, self.N_num:]] if self.N_cat > 0 else X_valid
        
        # train the model
        self.history = self.model.fit(
            x=X_train,
            y=Y_train,
            batch_size=self.batch_size,
            epochs=self.max_epochs,
            validation_data=(X_valid, Y_valid),
            # steps_per_epoch=self.steps_per_epoch,
            verbose=1
        )
        self.save()
        self.plot_history(log_scale=False)
        self.evaluate_model(X_valid, Y_valid)
        # self.clear_memory(X_train, X_valid, Y_train, Y_valid)
        return self.model, self.history
    
    
    def clear_memory(self, X_train, X_valid, Y_train, Y_valid):
        """Clear the Keras session to free up memory."""
        if self.model is None:
            self.logger.info("No model to clear.")
            return
        tf.keras.backend.clear_session()
        self.model = None
        self.logger.info("Keras session cleared and model set to None.")
        used_memory_before_MB = psutil.Process().memory_info().rss / (1024 * 1024)
        del X_train, X_valid, Y_train, Y_valid
        used_memory_after_MB = psutil.Process().memory_info().rss / (1024 * 1024)
        self.logger.info(f'memory used before cleaning: {used_memory_before_MB:.2f} MB')
        self.logger.info(f'memory used after cleaning: {used_memory_after_MB:.2f} MB')
        self.logger.info(f'difference: {used_memory_before_MB-used_memory_after_MB:.2f} MB')



    def predict(self, X, Y=np.array([])):
        """ 
        Predict the target data using the trained model.
        If Y is provided, it is used for evaluation.
        """
        
        self.normalize_numerical_arrays(X, Y, scalers=self.scalers)  
    
        #TBD!!!! transfer this to array prep (before saving!)
        Y = CNNModel._patch_middle_value(Y)
                
        # number of samples to be fed to model at once
        chunk_size = 20000  # TBD!!!! hardcoded
        n_patches = X.shape[0]
        num_chunks = int(np.ceil(n_patches / chunk_size))  
        self.logger.info(f'    Dividing the prediction dataset in {num_chunks} chunks of size {chunk_size}')

        # derive indices for each chunk
        chunks_indices = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_patches)
            chunks_indices.append((start_idx, end_idx))
                
        # run model.predict() on chunks 
        Y_hat= np.array([])
        self.logger.info('    Running model.predict() on chunks ...')
        for i,idx in enumerate(chunks_indices):
            if (i+1)%10==0:
                self.logger.info(f'        {i+1}/{len(chunks_indices)}')
            X_chunk = X[idx[0]:idx[1], :, :, :]
            X_chunk = [X_chunk[:, :, :, :self.N_num], X_chunk[:, :, :, self.N_num:]] if self.N_cat > 0 else X_chunk
            Y_hat_chunk = self.model.predict(X_chunk, verbose=0)
            Y_hat = np.concatenate([Y_hat,Y_hat_chunk]) if len(Y_hat)>0 else Y_hat_chunk
            del X_chunk, Y_hat_chunk
        
        # rescale Y_hat to original scale   
        Y_hat = self.scalers[self.target_dataset].inverse_transform(Y_hat)
        return Y_hat
            
            
            
            







       
        


# def cnn_model(input_shape, num_numerical_features, normalize, dropout, dropout_rate, steps_per_epoch, no_data_value):
    
#     # Numerical branch with BatchNormalization
#     numerical_input = Input(shape=(input_shape[0], input_shape[1], num_numerical_features))
#     if normalize:
#         x_numerical = BatchNormalization()(numerical_input)
#     x_numerical = Conv2D(128, kernel_size=(2, 2), strides=(1, 1), activation='relu')(x_numerical)
#     x_numerical = Conv2D(128, kernel_size=(2, 2), strides=(1, 1), activation='relu')(x_numerical)
#     if normalize:
#         x_numerical = BatchNormalization()(x_numerical)
#     if dropout:
#         x_numerical = Dropout(dropout_rate)(x_numerical)

#     x_numerical = Conv2D(512, kernel_size=(2, 2), strides=(1, 1), activation='relu')(x_numerical)
#     x_numerical = Conv2D(512, kernel_size=(2, 2), strides=(1, 1), activation='relu')(x_numerical)
#     if normalize:
#         x_numerical = BatchNormalization()(x_numerical)
#     if dropout:
#         x_numerical = Dropout(dropout_rate)(x_numerical)

#     x_numerical = Flatten()(x_numerical)
    
    
#     # Categorical branch (no BatchNormalization for categorical features)
#     num_categorical_features = input_shape[2] - num_numerical_features
    
#     if num_categorical_features > 0:
        
#         categorical_input = Input(shape=(input_shape[0], input_shape[1], num_categorical_features))
        
#         x_categorical = Conv2D(128, kernel_size=(2, 2), strides=(1, 1), activation='relu')(categorical_input)
#         x_categorical = Conv2D(128, kernel_size=(2, 2), strides=(1, 1), activation='relu')(x_categorical)
#         if dropout:
#             x_categorical = Dropout(dropout_rate)(x_categorical)

#         x_categorical = Conv2D(512, kernel_size=(2, 2), strides=(1, 1), activation='relu')(x_categorical)
#         x_categorical = Conv2D(512, kernel_size=(2, 2), strides=(1, 1), activation='relu')(x_categorical)
#         if dropout:
#             x_categorical = Dropout(dropout_rate)(x_categorical)

#         x_categorical = Flatten()(x_categorical)
        
#         # Concatenate the outputs of the numerical and categorical branches
#         merged = Concatenate()([x_numerical, x_categorical])
        
#     else:
        
#         # If no categorical features, just use the numerical features
#         merged = x_numerical
        
        
        

#     # Dense layers and output
#     x = Dense(128, activation='relu')(merged)
#     if dropout:
#         x = Dropout(dropout_rate)(x)

#     output = Dense(1, activation='linear')(x)


#     # Create the model with appropriate inputs
#     if num_categorical_features > 0:
#         model = Model(inputs=[numerical_input, categorical_input], outputs=output)
#     else:
#         model = Model(inputs=numerical_input, outputs=output)

#     # Configure optimizer
#     lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#         initial_learning_rate=1e-4,
#         decay_steps=steps_per_epoch * 1000,
#         decay_rate=1,
#         staircase=False)
#     optimizer = Adam(learning_rate=lr_schedule)
#     # optimizer = Adam(learning_rate=0.0001)

#     # Create a partial function with the fixed no_data_value argument
#     custom_loss_function = partial(ecostress_custom_loss, no_data_value=no_data_value)

#     model.compile(loss=custom_loss_function, optimizer=optimizer)
#     # model.compile(loss=ecostress_custom_loss, optimizer=optimizer, metrics=['loss', 'accuracy'])

    
        
#     # model = Sequential()
#     # if normalize:
#     #     model.add(BatchNormalization(input_shape=input_shape))
    
#     # # CNN
#     # model.add(Conv2D(128, kernel_size=(2, 2), strides=(1, 1), input_shape=input_shape, activation='relu'))
#     # model.add(Conv2D(128, kernel_size=(2, 2), strides=(1, 1), activation='relu'))
#     # if normalize:
#     #     model.add(BatchNormalization())
#     # if dropout:
#     #     model.add(Dropout(dropout_rate))

#     # model.add(Conv2D(512, kernel_size=(2, 2), strides=(1, 1), activation='relu'))
#     # model.add(Conv2D(512, kernel_size=(2, 2), strides=(1, 1), activation='relu'))
#     # if normalize:
#     #     model.add(BatchNormalization())
#     # if dropout:
#     #     model.add(Dropout(dropout_rate))

#     # model.add(Flatten())

#     # #Dense 1
#     # model.add(Dense(128, activation='relu'))
#     # if normalize:
#     #     model.add(BatchNormalization())
#     # if dropout:
#     #     model.add(Dropout(dropout_rate))

#     # # Output 
#     # model.add(Dense(1, activation='linear'))

#     # # configure optimizer
#     # lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#     #     initial_learning_rate = 1e-3,
#     #     decay_steps = steps_per_spoch * 1000 ,
#     #     decay_rate = 1,
#     #     staircase = False )
#     # optimizer = Adam(learning_rate=lr_schedule)
#     # # optimizer = Adam(learning_rate=0.0001, decay=0.0)
   
#     # model.compile(loss=ecostress_custom_loss, optimizer=optimizer)  #, metrics=['accuracy'])
    
#     return model
