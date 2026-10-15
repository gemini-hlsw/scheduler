Fix issues with ODB event handler in Night monitor:
- The path for created was throwing a TypeError. That path is unlikely as most addition are handle by Defined -> Ready path
- On deleted observations the reference label was incorrectly being matched with ObservationID.
- ObservationStatus stored in the last plan was being modified by GM instead of compared with the real ODB retrieved data.
- IDLE Timer had the wrong length for partial observations. 
