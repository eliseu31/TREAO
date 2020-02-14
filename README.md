# Task/Resource Allocation Simulator 

## Installation

### Usage

```bash
# display the help message
python simulation/main.py -h

# fit and display the machine distributions
python simulation/main.py -p resources/profiling_results.csv -f resources/function_req.json -d rasp_pi

# simulates different types of pipelines configurations
python simulation/main.py -s resources/sim_place.json

# optimize the graph pipeline
# using the specified parameters
# and producing the report in the defined json
python simulation/main.py -g resources/graph.json -o resources/opt_param.json resources/opt_report.json
```
