#ifndef _RING_BUFFER_H_
#define _RING_BUFFER_H_

template <class T>
class RingBuffer
{
    public:

        explicit RingBuffer(int size)
        : Size(size),
          Buffer(std::vector<T>())
        {
        }

        ~RingBuffer() { }

        // there is probably a more efficient way of doing this with InPtrs etc. I couldn't immediately think of how to do it and (simply) satisfy the calling fns
        // use of end() vector iterators. 
        void push_back(T &item)
        {
            if (Buffer.size() < Size)
            {
                Buffer.push_back(item);
            }
            else 
            {
                for (int i = 0; i < Size - 1; ++i)
                {
                    Buffer[i] = Buffer[i + 1];
                }
                Buffer[Size - 1] = item;                
            }
        }

        int size() const { return Buffer.size(); }
        typename std::vector<T>::iterator end() { return Buffer.end(); }
        typename std::vector<T>::const_iterator end() const { return Buffer.end(); }

    private:

        int Size;
        typename std::vector<T> Buffer;

};


#endif