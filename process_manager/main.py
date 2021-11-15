from __init__ import ProcessManager

if __name__ == "__main__":
    
    config = {
        'semestral_bins': {
            'amount': 1,
            'bin_size': 2,
            'n_threads': 2
        },
        'week_bins': {
            'amount': 1,
            'bin_size': 2,
            'n_threads': 2
        },
        'custom_bins': {
            'amount': 1,
            'bin_size': 2,
            'n_threads': 2
        },
        'realtime_bins': {
            'amount': 1,
            'bin_size': 2,  # does make sense to have more than one bin for realtime mode?
            'n_threads': 1
        }
        
    }

    manager = ProcessManager(config)
    manager.run()
