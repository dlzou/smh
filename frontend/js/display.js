export let data = [];

// export const COLORS = {
//     "water" : "blue",
//     "gun": "red",
// }

export function createRow(type) {
    let currTime = new Date();
    data.push({
        type: type,
        time: (currTime.getFullYear()
        + "/" + (currTime.getMonth() + 1) 
        + "/" + currTime.getDay() 
        + " " + currTime.getHours()
        + ":" + currTime.getMinutes())
    })
    print(data);
}

export function getData() {
    return fetch('http://bigrip.ocf.berkeley.edu:5000/getdata')
        .then(response => response.json())
        .then(rawData => {
            data = rawData;
        })
        .catch(error => {
            console.error(error);
        });
}