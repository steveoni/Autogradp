// var utils = require("./utils.js");

var autograd = {};

(function(global){

    function makeid(length) {
        var result           = '';
        var characters       = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
        var charactersLength = characters.length;
        for ( var i = 0; i < length; i++ ) {
            result += characters.charAt(Math.floor(Math.random() * charactersLength));
        }
        return result;
     }

    var Mat = function (n, d) {
        // n is number of rows d is number of columns
        this.n = n;
        this.d = d;
        this.out = utils.zeros(n * d);
        this.dout = utils.zeros(n * d);
    }

    var Tensor = function (n, d, require_grad) {

        this.n = n;
        this.d = d;
        this.out = utils.zeros(n * d);
        this.dout = utils.zeros(n * d);
        this.require_grad = require_grad;
        this.name = null;
    }

    Tensor.prototype = {

        get: function (row, col) {
    
            var ix = (this.d * row) + col;
            utils.assert(ix >= 0 && ix < this.out.length);
            return this.out[ix];
    
        },
        set: function (row, col, v) {
            var ix = (this.d * row) + col;
            utils.assert(ix >= 0 && ix < this.out.length);
            this.out[ix] = v;
        },
        setFrom: function (arr) {
            utils.assert(arr.length == this.n*this.d,"shape not compatible")
            for (var i = 0, n = arr.length; i < n; i++) {
                this.out[i] = arr[i];
            }
        },
        randn: function (mu, std) {
            utils.fillRandn(this.out, mu, std);
            return this;
    
        },
        grad: function (grad) {
    
            // for(var i=0,n=grad.length;i<n;i++){
            //     this.dout[i] = grad[i];
            // }
            this.dout = grad;
        }
    }

    function add(x, y) {
        utils.assert(x.out.length === y.out.length);
    
    
        this.items = new Mat(1,x.out.length);
        for (var i = 0; i < x.out.length; i++) {
    
            this.items.out[i] = x.out[i] + y.out[i];
        }
        this.x = x;
        this.y = y;
        this.require_grad = true;
        this.out = this.items.out;
        this.dout = this.items.dout;
        this.n = this.items.n;
        this.d = this.items.d;
        this.func_name = "<add>";
    
        // this.gradv = 1;
    
    }

    add.prototype = {

        backward: function () {
    
            if (this.x.require_grad) {
                this.x.grad(this.dout);
                if ("backward" in this.x) {
                    this.x.backward()
                }
    
    
            }
    
            if (this.y.require_grad) {
                this.y.grad(this.dout);
                if ("backward" in this.y) {
                    this.y.backward()
                }
            }
        },
    
        grad: function (g) {
    
            utils.assert(this.items.dout.length === g.length);
            this.dout = g;
    
        }
    }

    function Matmul(x, y) {

        utils.assert(x.d === y.n, "matmul dimension misaligned");
    
        this.n = x.n;
        this.d = y.d;
        this.x = x;
        this.y = y;
        this.require_grad = true;
        this.items = new Mat(this.n, this.d);
        this.out = this.items.out;
        this.dout = this.items.dout
        this.func_name = "<Multiply>";
    
        for (var i = 0; i < x.n; i++) {
            for (var j = 0; j < y.d; j++) {
    
                var dot = 0.0;
                for (var k = 0; k < x.d; k++) {
    
                    dot += this.x.out[x.d * i + k] * this.y.out[y.d * k + j];
                }
                this.out[this.d * i + j] = dot;
            }
        }
    }

    Matmul.prototype = {

        backward: function () {
    
            if (this.x.require_grad) {
                
                for(var i = 0;i< this.x.n;i++){
                    for(var j=0;j<this.y.d;j++){
                        for(var k =0;k<this.x.d;k++){
                            var b = this.dout[this.y.d*i+j];
                            // console.log(b);
                            this.x.dout[this.x.d*i+k] += this.y.out[this.y.d*k+j] * b;
    
                        }
                    }
                }
    
                if ("backward" in this.x) {
                    this.x.backward()
                }
    
    
            }
    
            if (this.y.require_grad) {
    
                for(var i = 0;i< this.x.n;i++){
                    for(var j=0;j<this.y.d;j++){
                        for(var k =0;k<this.x.d;k++){
                            var b = this.dout[this.y.d*i+j];
                            this.y.dout[this.y.d*k+j] += this.x.out[this.x.d*i+k] * b;
    
                        }
                    }
                }      
    
                if ("backward" in this.y) {
                    this.y.backward()
                }
            }
        },
    
        grad: function (g) {
    
            utils.assert(this.dout.length === g.length);
            this.dout = g;
    
        }
    }

    function Linear(in_dim,out_dim){

        this.W = new Tensor(in_dim,out_dim,true).randn(0,0.008);
        this.b = new Tensor(out_dim,1,true).randn(0,0.008);
    
        this.func_name = "<Linear>";
        this.require_grad = true;
    
    }

    Linear.prototype = {

        forward : function(x){
    
            this.mult = new Matmul(x,this.W)
            this.items = new add(this.mult,this.b);
            this.n = this.mult.n;
            this.d = this.mult.d;
            this.out = this.items.out;
            this.dout = this.items.dout;
            
    
            return this;
        },
    
        backward: function(){
    
                this.items.grad(this.dout);
                this.items.backward();
        },
        grad: function(g){
            utils.assert(this.dout.length === g.length);
            this.dout = g;
        },
    
        update: function(lr){
    
            for(var i=0;i< this.W.out.length;i++){
    
                this.W.out[i] -= (lr*this.W.dout[i]); 
            }
            for(var i=0;i< this.b.out.length;i++){
    
                this.b.out[i] -= (lr*this.b.dout[i]); 
            }
    
    
    
        },

    }
    
    function ReLU(){

        this.require_grad = true;
        this.func_name = "<ReLu>";
    
    }

    ReLU.prototype = {

        forward : function(x){
            this.items = new Mat(x.n,x.d);
            this.x = x;
            this.n = x.n;
            this.d = x.d;
            
            for(var i=0;i<x.out.length;i++){
    
                this.items.out[i] = Math.max(0,x.out[i]);
            }
            this.out = this.items.out;
            this.dout = this.items.dout;
    
            return this;
        },
    
        backward: function(){
            
            
            for(var i=0;i<this.x.out.length;i++){
    
                this.x.dout[i] = this.x.out[i] > 0 ? this.dout[i] : 0.0;
            }
            this.x.backward();
    
    
        },
        grad: function(g){
            utils.assert(this.dout.length === g.length);
    
            this.dout = g;
    
        }
    }

    function Softmax(){
        this.func_name = "<Softmax>";
    }

    Softmax.prototype = {

        forward : function(x){
            this.items = new Mat(1,x.d)
    
            this.x = x;
            //compute max activation
            var as = x.out;
            var amax = x.out[0];
            for(var i=1;i<this.x.d;i++){
    
                if(as[i] > amax) amax = as[i];
            }
    
            var es = utils.zeros(this.x.d);
            var esum = 0.0;
            for(var i=0;i<this.x.d;i++){
    
                var e = Math.exp(as[i] - amax);
                esum += e;
                es[i] = e;
            }
    
            //normalize and output to sum one
            for(var i=0;i<this.x.d;i++){
                es[i] /= esum;
                this.items.out[i] = es[i];
            }
            this.out = es; // saved for backprop
    
            return this;
        },
    
        backward: function(y){
    
            var x = this.x;
            for(var i=0;i<this.items.d;i++){
    
                var indicator = i === y ? 1.0 : 0.0;
                var mul = -(indicator - this.out[i])
                x.dout[i] = mul;
            }
    
            this.x.grad(x.dout)//there is no need for this. we initiallizing dw twice
            this.x.backward();
        }
    }

    function Sequential(models){

        this.models = models;

        this.weights = []

        let w_count = 0;
        for(let i=0;i<this.models.length;i++){

            this.models[i].func_name = `${this.models[i].func_name}_${i}`

            if(this.models[i].hasOwnProperty("W") && this.models[i].hasOwnProperty("b")){
                console.log("out there")
                this.models[i].W.name = `weight_${w_count}`
                this.models[i].b.name = `bias_${w_count}`

                this.weights.push(this.models[i].W.name)
                this.weights.push(this.models[i].b.name)

                w_count +=1
            }

        }
    }

    Sequential.prototype = {

        forward : function(x){
    
            
            this.out = this.models[0].forward(x);
            

            for(var i=0;i<this.models.length;i++){
    
                if( i==0){
                    continue;
                }
                this.out = this.models[i].forward(this.out);
                if(this.out.hasOwnProperty("mult") && this.out.hasOwnProperty("items")){
                    console.log("in here");
                    this.out.mult.func_name = `${this.out.mult.func_name}_${i}`;
                    this.out.items.func_name = `${this.out.items.func_name}_${i}`;
                }
            }
        },
    
    }

    function Loss(target, predict){

        this.model = predict.out;
        this.out =  - Math.log(predict.out.out[target]);;
        this.y = target;
    }

    Loss.prototype = {

        backward : function(){
    
            this.model.backward(this.y);
            
        }
    }

    function OptimSGD(model,lr){

        this.model = model;
        this.lr = lr;
    
    }
    
    OptimSGD.prototype = {

        step : function(){
    
            for(var i in this.model.models){
                
                // console.log(model)
                if("update" in this.model.models[i]){
                    // console.log("here")
                    this.model.models[i].update(this.lr);
                }
            }
        },
        grad_zero:  function(){

            for(var i in this.model.models){
                
                // console.log(model)
                if(Object.prototype.hasOwnProperty.call(this.model.models[i],"W")){
                    // console.log("here")
                    let len_w = this.model.models[i].W.out.length
                    let len_b =  this.model.models[i].b.out.length

                    this.model.models[i].W.dout = utils.zeros(len_w);
                    this.model.models[i].b.dout = utils.zeros(len_b);
                }
            }

        }
    }

    global.Mat = Mat;
    global.Tensor = Tensor;
    global.add = add;
    global.matmul = Matmul;
    global.Linear = Linear;
    global.ReLU = ReLU;
    global.Softmax = Softmax;
    global.Sequential = Sequential;
    global.Loss = Loss;
    global.OptimSGD = OptimSGD;

})(autograd)

// var model = new autograd.Sequential([
//     new autograd.Linear(2,3),
//     new autograd.ReLU(),
//     new autograd.Linear(3,2),
//     new autograd.Softmax()
// ]);

// var x = new autograd.Tensor(1,2,require_grad=true)
// x.setFrom([2,3]);

// x.name = "input"

// var y = new autograd.Tensor(2,4,require_grad=false);
// y.setFrom([2,3,4,5,6,7,8,1])

// var m =model.forward(x)

// var l = new autograd.Loss(1,model.out)
// console.log(model.models[3], l.out)
// l.backward()
// console.log(model.models[0].dout, l.out)
// model.forward(x)


//  let linear1 =    new autograd.Linear(2,3)
//  let relu =    new autograd.ReLU()
//  let linear2 =   new autograd.Linear(3,2)
//  let softmax =    new autograd.Softmax()


// var x = new autograd.Tensor(1,2,require_grad=true)
// x.setFrom([2,3]);


// input = linear1.forward(x)
// input = relu.forward(input)
// input =  linear2.forward(input)
// // output = softmax.forward(input);

// console.log(model.weights)

// softmax.backward(1)

// console.log(output.dout,linear1.dout)

// console.log(JSON.stringify(output))

function model_get(json,value){

    return json[value]
}



// console.log(Network(model.out))