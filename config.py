"""
Adithya Bhaskar, 2022
This file contains all the configuration options to be used by the Reverse Dictionary
model. 
"""
force_download = False                      # Force re-download of data
process_dictionaries=True                   # Prune dictionaries to remove low-quality data
filter_using_glove = True                   # Filter to words contained in glove. Must be set to True.
dont_use_edmt = True                        # Don't use the EDMT data
dont_use_webster = True                     # Don't use the Webster data
dont_use_wordnet = False                    # Don't use wordnet
dont_use_unix = False                       # Don't use the unix+vocabulary.com data
use_multi_layers = False                    # Use the MultiLayerLSTM model
save = True                                 # Save model after each epoch
force_restart_training = False              # Force training to restart from scratch
NUM_TARGET_EPOCHS = 50                      # Target number of epochs, used for linear schedule
NUM_EPOCHS = 50                             # Number of epochs for this run.
BATCH_SIZE = 32                             # Batch size.
CHECKPT_DIR = "checkpoints/"                # Directory with checkpoints.
                                            #  Change if you change use_multi_layers !
remove_bad_words = True                     # Whether to not consider words in sample_bad_words
                                            #  when decoding. This was set to False while computing
                                            #  the reported metrics, but improves QoL.
sample_bad_words = ['timewrn', 'svahng',   \
    'bulletinyyy', 'seabream', 'srivalo',  \
    'nortelnet', 'piyanart', 'prohertrib', \
    'canyonres']                            # Some most common 'worng' words that are sometimes 
                                            # in top-1