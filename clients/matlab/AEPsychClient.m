% Copyright (c) Facebook, Inc. and its affiliates.
% All rights reserved.

% This source code is licensed under the license found in the
% LICENSE file in the root directory of this source tree.

classdef AEPsychClient
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
            % Send the server a configuration string detailing the
            % experiment to be run
            config_long_string = sprintf('%s', config{:});
            config_msg = sprintf('{"type":"setup","version":"0.01","message":{"config_str":"%s"}}', config_long_string);
            response = self.send_recv(config_msg);
            self.strat_indices(end+1) = str2num(response);
        end
        function response=ask(self)
            % Request from the server the next trial configuration to be
            % run
            ask_msg = '{"type":"ask","message":""}';
            response = jsondecode(self.send_recv(ask_msg));
            % jsondecode destroys singleton arrays which we don't like, so
            % we convert them to cell arrays which retains
            fn = fieldnames(response);
            for k = 1:numel(fn)
                response.(fn{k}) = num2cell(response.(fn{k}));
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
            fprintf("Requested strat %d, got strat %s", strat_id, response);
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
