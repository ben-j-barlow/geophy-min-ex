# Function to generate the log file name based on the current datetime
function generate_log_file_name()
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    return "logging/log_$timestamp.log"
end

function prepare_logger()
    name = generate_log_file_name()
    io = open(name, "w+")
    logger = SimpleLogger(io)
    global_logger(logger)
    return io
end

function close_logger(io::IOStream)
    close(io)
end