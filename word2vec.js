var utils = require("./utils.js");
var nn = require("./autograd.js")

let text = "The king is a man who rules over a nation, he always have a woman beside him called the\
 queen.\n she helps the king controls the affars of the nation.\n Perhaps, she acclaimed the position of a king\
 when the king her husband is deceased."


text_lower = text.toLocaleLowerCase()

text_list = text_lower.split("\n")

var stopwords = ["a","in","when","the","of","is","who"]

function clean(text){

    let textt = text.split(' ')

    //filter out empty string
    let text_filter = textt.filter((val)=>{
            return val !=''
    });

    let stop_wordFilter = text_filter.filter((val)=>{
            
            return !stopwords.includes(val);
    });

    let puntionless = stop_wordFilter.map((val)=>{

         return val.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g,"")
    });

    return puntionless;
}

// console.log(clean(text_list[0]))

function gen_word(window=5, text_list){

    let word_list = []
    let all_text = []

    for(let index in text_list){

        let text = text_list[index];

        let text_clean = clean(text);

        all_text.push(...text_clean);


        for(let i=0;i <text_clean.length; i++){

            let word = text_clean[i]
            for(let j=0;j < window; j++){

                if((i + 1 + j) < text_clean.length){
                    word_list.push([word, text_clean[i+1+j]])
                }

                if((i-j-1) >=0){
                    word_list.push([word, text_clean[i-j-1]])
                }
            }
        }
        
    }

    return [word_list, all_text]
}

let [word_list, all_text] = gen_word(5,text_list);


function unique_word(text_list){

    let word_set = new Set(text_list);

    let word_list = Array.from(word_set);

    let unique_word  = {}

    for(let i=0; i < word_list.length; i++){
        
         let word = word_list[i]
         unique_word[word] = i+1;
    }

    return unique_word
}

let unique_dict = unique_word(all_text)

function obj_len(dict){

    let count = 0

    for(let i in dict){
        count +=1
    }
    return count;
}

let n_words = obj_len(unique_dict);

console.log(n_words);


function create_data(word_list){

    let  data = []
    let label = []

    for(let i=0; i< word_list.length;i++){

        let x = word_list[i][0]
        let y = word_list[i][1]

        let word_index = unique_dict[x]
        let context_index = unique_dict[y]

        let X_row = utils.zeros(n_words)
        // let y_row = utils.zeros(n_words)

        X_row[word_index] = 1.

        // y_row[word_index] = 1.

        data.push(X_row)
        label.push(context_index)
    }

    return [data, label];
}

let [data, label] = create_data(word_list)


let embed_dim = 50;
let model = new nn.Sequential([
        new nn.Linear(n_words,embed_dim),
        new nn.Linear(embed_dim,n_words),
        new nn.Softmax()
]);

let optim = new nn.OptimSGD(model,lr=0.001);

// let x = new nn.Tensor(1,n_words, true);
// x.setFrom(data[0])

// let m  = model.forward(x);

// console.log(label);

epoch = 50
for(let i=0; i< epoch; i++){

    let total_loss = 0;
    for(let j=0; j < data.length; j++){

        let x_data = data[j]
        let y_data = label[j]

        let x = new nn.Tensor(1,n_words, false);
        x.setFrom(x_data)

        model.forward(x)

        // console.log(-Math.log(model.out.out[y_data-1]))
        let loss = new nn.Loss(y_data-1,model)

        // console.log(loss.out);
        total_loss += loss.out

        loss.backward()

        optim.step();

        optim.grad_zero()

    }

    console.log(`for epoch ${i} Loss is ${total_loss/data.length}`)
}


function get_weight(weight){

    let row = weight.n

    let col = weight.d

    let data = Array(row);

    for(var i=0;i<row;i++){
        data[i] =[]
        for(var j=0;j<col;j++){
            /**
             * since the array are store in this form [1,2,3,4,5,6,7,8,9,10,11,12]
             * such an array suppose to be a 2d array [[1,2,3,4,5,6],[7,8,9,10,11,12]]
             * in which the first array in the 2d array is 0
             * the best way to specifiy the index in the 1d array is:
             * (arr.length*jth_col).column.length + ith_row
             */
            var indices = (weight.out.length*j)/col + i;
            
            data[i].push(weight.out[indices]);
        }
    }
    
    return data


}

let embed_weight = get_weight(model.models[0].W)
console.log(embed_weight[0].length)
