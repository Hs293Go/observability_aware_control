
%% Define the state vector
r_lf = sym("r", [3, 1], 'real'); % Position of leader w.r.t. follower in follower's frame
q_fl = sym("q", [4, 1], 'real'); % Orientation of follower relative to leader
v_lf = sym("v", [3, 1], 'real'); % Velocity of leader w.r.t. follower in follower's frame
x = [r_lf; q_fl; v_lf];

%% Define the input vector
f_l = sym("f_l", "real");
omega_l = sym("omega_l", [3, 1], "real");
f_f = sym("f_f", "real");
omega_f = sym("omega_f", [3, 1], "real");

%% Full state dynamics equation (for verification)
dx = [-cross(omega_f, r_lf) + v_lf; ...
    -QuaternionProduct([omega_f / 2; 0], q_fl) + QuaternionProduct(q_fl, [omega_l / 2; 0]);
    -cross(omega_f, v_lf) + QuaternionRotatePoint(q_fl, [0; 0; f_l]) - [0; 0; f_f];
    ];

%% Control-affine dynamics equation
R = QuaternionToRotationMatrix(q_fl);
QL = 1 / 2 * [q_fl(4) * eye(3) + Hat(q_fl(1:3)); -q_fl(1:3).'];
QR = 1 / 2 * [q_fl(4) * eye(3) - Hat(q_fl(1:3)); -q_fl(1:3).'];
f = cell(5, 1);
f{1} = [v_lf; zeros(7, 1)];
f{2} = [zeros(7, 1); QuaternionRotatePoint(q_fl, [0; 0; 1])];
f{3} = [zeros(3); QL; zeros(3)];
f{4} = [zeros(7, 1); 0; 0; -1];
f{5} = [Hat(r_lf); -QR; Hat(v_lf)];

% Check the control-affine form is equivalent to the full state dynamics
assert(all(simplify(dx == f{1} + f{2} * f_l + f{3} * omega_l + f{4} * f_f + f{5} * omega_f )));


%% Define observation model
nr = sqrt(r_lf.'*r_lf);
h = [r_lf.' * r_lf / 2; q_fl];

% Zeroth order
Lh = h;
DLh = jacobian(Lh, x);

% First order
Lhf0 = DLh * f{1};
DLhf0 = jacobian(Lhf0, x);

% Second order by 0
L2hf0 = DLhf0 * f{1};
DL2hf0 = jacobian(L2hf0, x);

L2hf01 = DL2hf0 * f{2};
DL2hf01 = jacobian(L2hf01, x);

Lhf01 = DLhf0 * f{2};
DLhf01 = jacobian(Lhf01, x);

Lhf03 = DLhf0 * f{4};
DLhf03 = jacobian(Lhf03, x);

O = [DLh; 
    DLhf0; 
    DL2hf0; 
    DLhf01;
    DLhf03;
    DL2hf01];
O( all( isAlways(O==0) ,2) ,:) = [];
i3 = [0; 0; 1];
Jq = 2 * [q_fl(1:3).' * i3 * eye(3) - Hat(cross(q_fl(1:3), i3)) - q_fl(4) * Hat(i3), cross(q_fl(1:3), i3) + q_fl(4) * i3];
fprintf("Rank of observation matrix is: %d\n", rank(O));

SO = O;
SO(2:5, :) = [];
SO(:, 4:7) = [];

%%

Lhf = DLh * dx;
DLhf = jacobian(Lhf, x);
L2hf = DLhf * dx;
DL2hf = jacobian(L2hf, x);
L3hf = DL2hf * dx;
