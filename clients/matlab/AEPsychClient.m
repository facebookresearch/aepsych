% Copyright (c) Meta Platforms. and its affiliates.
% All rights reserved.

% This source code is licensed under the license found in the
% LICENSE file in the root directory of this source tree.

classdef AEPsychClient < handle
% AEPsychClient  Client to AEPsych
%   client = AEPsychClient() instantiates with the default params
%   (localhost:5555)
%   Optional args:
%     server_addr: address of AEPsych server
%     port: port of AEPsych server
%     timeout: how long (in seconds) to wait on connection or response
%     before bailing
    properties
        server_addr = 'localhost';  % address to connect on
        port = 5555;  % port to connect on
        timeout;  % timeout, how long to wait before bailing on connection/response
        connection;  % tcp connection object
        strat_indices = [];  % indices of strats we created
        is_finished = false;

    end
    methods
        % constructor
        function self = AEPsychClient(varargin)
            % this creates a connection to the server (but doesn't do
            % anything else)
            p = inputParser;
            addOptional(p, 'server_addr', 'localhost');
            addOptional(p, 'port', 5555);
            addOptional(p, 'timeout', 10);
            parse(p, varargin{:});
            self.server_addr = p.Results.server_addr;
            self.port = p.Results.port;
            self.timeout = p.Results.timeout;
            self.connection = tcpclient(self.server_addr, self.port, "Timeout", self.timeout);

        end
        % destructor
        function delete(self)
            % destructor, remove the connection
            delete(self.connection);
            clear self.connection;
        end
        % public API
        function configure(self, config)
            %Send a config to set up the server
            config_long_string = sprintf('%s', config{:});
            self.setup(config_long_string);
        end
        
        function configure_by_file(self, filename, metadata_dict)
            % Read a file into a string to send to the server
            if nargin < 3
                metadata_dict='{}';
            end
            fid=fopen(filename);
            config_string='';
            tline = fgetl(fid);
            while ischar(tline)
                config_string = strcat(config_string, tline,'\n'); 
                tline = fgetl(fid);
            end
            fclose(fid);
            config_string = strrep(config_string, '"', '\"');
            self.setup(config_string, metadata_dict);
        end
        
        function setup(self, config_string, metadata_dict)
            % Send the server a configuration string detailing the
            % experiment to be run
            if nargin < 3
                metadata_dict='{}';
            end
            config_msg = sprintf('{"type":"setup","version":"0.01","metadata":%s,"message":{"config_str":"%s"}}', metadata_dict, config_string);
            response = self.send_recv(config_msg);
            self.strat_indices(end+1) = str2num(response);
        end
        
        function [fmax,loc] = get_max(self)
            % Get the model maximum point and its location 
            msg = sprintf('{"type":"query", "message":{"query_type":"max"}}');
            response = jsondecode(self.send_recv(msg));
            loc = response.x;
            fmax = response.y;    
        end
            
        
       function [prob,loc] = find_val(self, val, prob_space)
           % Find a point in the model closest to the given val val.
           % If prob_space is true, input a probability and this function
           % will find a point in the model with that probability.
           msg = struct("type","query", "message", struct("query_type","inverse", "probability_space", prob_space, "y",val));
           response = self.send_recv(jsonencode(msg));
           response = jsondecode(response)
           loc = response.x;
           prob = response.y;    
       end
       
       
       function [fval, loc] = predict(self, config, prob_space)
           % Model predict at the location given in the {param : value} dict
           % defined in the in config
           msg = struct("type", "query", "message", struct("x",config, "query_type","prediction", "probability_space", prob_space));
           response = self.send_recv(jsonencode(msg));
           response = jsondecode(response);
           loc = response.x;
           fval = response.y;
       end
       
       function can_model = get_can_model(self)
            msg = '{"type":"can_model","message":""}'; 
            full_response = jsondecode(self.send_recv(msg));
            can_model = full_response.can_model;
       end
        
       function response=ask(self)
            % Request from the server the next trial configuration to be
            % run
            ask_msg = '{"type":"ask","message":"", "version":"0.01"}';
            full_response = jsondecode(self.send_recv(ask_msg));
            % jsondecode destroys singleton arrays which we don't like, so
            % we convert them to cell arrays which retains
            self.is_finished = full_response.is_finished;
            fn = fieldnames(full_response.config);
            response = struct();
            for k = 1:numel(fn)
                response.(fn{k}) = num2cell(full_response.config.(fn{k}));
            end
        end
        
        function tell(self, config, outcome)
            % Report back to the server a trial configuration that was run
            % and an outcome (0 or 1). Note that this need not be the same
            % as a configuration previously received from the server.
            tell_msg = struct("type", "tell", "message", struct("config",config, "outcome", outcome));
            self.send_recv(jsonencode(tell_msg));
        end
        
        function resume(self, strat_id)
            % Resume a past strategy used in the current session,
            % corresponding to a different model and data. Each strategy is
            % fully independent: this is a way to interleave experiments or
            % different optimization runs. To use a composite strategy that
            % shares data, use SequentialStrategy in your configuration.
            ask_msg = sprintf('{"type":"resume","version":"0.01","message":{"strat_id":"%d"}}', strat_id);
            response = jsondecode(self.send_recv(ask_msg));
            fprintf("Requested strat %d, got strat %d\n", strat_id, response);
        end
    end
    
    % private methods
    methods (Access='private')
        function response=send_recv(self, msg)
%             flush(self.connection);
            writeline(self.connection, msg);
            fprintf("wrote %d bytes\n", self.connection.NumBytesWritten);
            % wait until we have a response
            waittime = 0;
            while waittime < self.timeout
                if self.connection.NumBytesAvailable > 0
                    break
                end
                pause(0.01);
                waittime = waittime + 0.01;
            end
            if waittime == self.timeout
                fprintf("Timed out waiting for response!");
                response = 'NULL';
            else
                response=char(read(self.connection));
                fprintf("Response is [%s]\n", response);
            end
        end


    end
end
