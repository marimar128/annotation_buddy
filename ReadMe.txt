1. Load the images you want to annotate into ImageJ, and combine them into a single hyperstack in default `tzcyx` order.

2. Save this hyperstack as a single '.tif' called 'annotate_me.tif', in this directory

3. Run `1_mister_clicky.py` to annotate your image data. Press the 's' key (frequently) to save your current annotations.

4. If desired, run `2_random_forest_clickbooster.py` to generate automatic annotations based on your manual annotations. Press the 'r' key in Mister Clicky to reload these annotations for viewing.

5. If desired, run `3_neural_network_clickbooster.py` to generate automatic annotations based on the random forest annotations. Press the 'r' key in Mister Clicky to reload these annotations for viewing.