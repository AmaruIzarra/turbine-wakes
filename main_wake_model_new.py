from data_config_new_wake import ConfigParser
import wake_model_new


# main file to run simulation

if __name__ == '__main__':

    params = ConfigParser().params

    wake_model_new.main(params)
















