import sys, getopt

command_input_arguments = sys.argv[1:] # exclude file name as input argument

try:
    opts, args = getopt.getopt(command_input_arguments, "", ["testing_with_fridge=", "running_on_nuc=", "verbose="])
    for opt, arg in opts:        
        if opt == "--testing_with_fridge":
            testing_with_fridge = arg == "true"
        elif opt == "--running_on_nuc":
            running_on_nuc = arg == "true"
        elif opt == "--verbose":
            verbose = arg == "true"            
except getopt.GetoptError:
    # print 'test.py -i <inputfile> -o <outputfile>'
    # sys.exit(2)
    print(" no args passed")